//! Picking a regional model from a location: `models regions --lat --lon` and
//! `models install --region auto` (#451).
//!
//! Most tests drive the binary against a seeded registry whose `geo-test` entry
//! has frozen region geometry, so the expected output does not move when the
//! published data does. One test reads the bundled `birdnet-plus-v30` geometry,
//! so a published-data change that moves a well-known city is noticed. The
//! installs stop at the agreement gate before any download, except the one that
//! serves its files from a loopback listener.
// Integration test crate: see `tests/model_manifest.rs` for why these are allowed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::time::{Duration, Instant};

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::{Value, json};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// The loopback server stops accepting after this long.
const SERVER_DEADLINE: Duration = Duration::from_secs(30);

/// Files one install downloads: the model and its labels.
const DOWNLOADS_PER_INSTALL: usize = 2;

/// What the agreement gate says when a non-interactive install has no `--yes`.
const GATE_REFUSAL: &str = "error: installing Geo Test requires agreement to acceptable_use \
    (https://example.org/AUP.md); no prompt can be shown without a terminal or under structured \
    output, so read each one and run the install again with --yes to agree\n";

/// A published region as the seeded registry carries it: `(slug, name, tier,
/// centroid, boxes, species, size in bytes)`.
type Tile = (
    &'static str,
    &'static str,
    u32,
    [f64; 2],
    &'static [[f64; 4]],
    usize,
    u64,
);

// Geometry copied from the published BirdNET+ V3.0.1 regions of the bundled
// registry (`birdnet-plus-v30`, registry version 9). Nordic and Baltics tie at
// tier 50; Western Palearctic is the broader tier 10 region around both.
const TILES: &[Tile] = &[
    (
        "nordic",
        "Nordic",
        50,
        [62.88, 17.685],
        &[[53.1, 72.66, 3.32, 32.05]],
        464,
        102_224_864,
    ),
    (
        "baltics",
        "Baltics",
        50,
        [56.65, 25.0],
        &[[53.5, 59.8, 20.0, 30.0]],
        479,
        102_547_724,
    ),
    (
        "western-palearctic",
        "Western Palearctic",
        10,
        [41.5, 14.0],
        &[[35.0, 71.0, -11.0, 45.0], [26.0, 40.0, -32.0, -13.0]],
        1446,
        123_361_440,
    ),
];

fn file(name: &str, base: &str, size: u64) -> Value {
    json!({
        "url": format!("{base}/{name}"),
        "filename": name,
        "size_bytes": size,
    })
}

/// The `geo-test` entry: the global model plus [`TILES`], an agreement-requiring
/// document, and every download pointed at `base`.
fn geo_entry(base: &str) -> Value {
    let mut variants = vec![json!({
        "id": "fp32",
        "classes": 11725,
        "model": file("global.onnx", base, 344_606_647),
        "labels": file("global-labels.txt", base, 470_040),
    })];
    for (slug, name, tier, centroid, boxes, classes, size) in TILES {
        variants.push(json!({
            "id": "fp32",
            "region": slug,
            "region_name": name,
            "group": "europe",
            "group_name": "Europe",
            "group_order": 0,
            "classes": classes,
            "model": file(&format!("{slug}.onnx"), base, *size),
            "labels": file(&format!("{slug}-labels.txt"), base, 10_000),
            "tier": tier,
            "centroid": {"lat": centroid[0], "lon": centroid[1]},
            "bboxes": boxes.iter().map(|b| json!({
                "lat_min": b[0], "lat_max": b[1], "lon_min": b[2], "lon_max": b[3],
            })).collect::<Vec<_>>(),
        }));
    }
    json!({
        "id": "geo-test",
        "name": "Geo Test",
        "description": "Seeded for the region lookup tests",
        "vendor": "Test",
        "version": "1.0",
        "model_type": "birdnet-v30",
        "license": {
            "type": "Apache-2.0",
            "url": "https://example.org/LICENSE",
            "commercial_use": true,
            "attribution_required": false,
            "share_alike": false,
        },
        "build": 1,
        // An empty selection map makes the family default the choice whatever
        // GPU libraries the host has, so the "Selected variant" line is fixed.
        "default_variant": "fp32",
        "selection": {},
        "variants": variants,
        "documents": [{
            "kind": "acceptable_use",
            "requires_agreement": true,
            "url": "https://example.org/AUP.md",
        }],
    })
}

/// `flat-test`: two regions and no geometry, like a registry published before
/// region boundaries.
fn flat_entry(base: &str) -> Value {
    let mut entry = geo_entry(base);
    entry["id"] = Value::from("flat-test");
    entry["name"] = Value::from("Flat Test");
    for variant in entry["variants"].as_array_mut().unwrap() {
        let variant = variant.as_object_mut().unwrap();
        variant.remove("tier");
        variant.remove("centroid");
        variant.remove("bboxes");
    }
    entry
}

/// Seed `home` with the bundled registry, newer than the bundled one so the
/// cached copy is kept, holding `geo-test` and `flat-test`. The geomodel is
/// dropped so the side install never reaches the network.
fn seed_registry(home: &Path, base: &str) {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("registry.json");
    let mut registry: Value =
        serde_json::from_slice(&std::fs::read(bundled).unwrap()).expect("bundled registry parses");
    registry["registry_version"] = Value::from(9999);
    registry.as_object_mut().unwrap().remove("range_filter");
    registry["models"] = json!([geo_entry(base), flat_entry(base)]);
    std::fs::write(home.join("registry.json"), registry.to_string()).unwrap();
}

fn write_location(home: &Path, lat: f64, lon: f64) {
    std::fs::write(
        home.join("config.toml"),
        format!("[defaults]\nlatitude = {lat}\nlongitude = {lon}\n"),
    )
    .unwrap();
}

struct Ran {
    status: i32,
    stdout: String,
    stderr: String,
}

/// Run `birda` in an isolated home with a null stdin and no analyze location in
/// the environment, adding `env` on top.
fn run_with(home: &Path, env: &[(&str, &str)], args: &[&str]) -> Ran {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
        .env_remove("BIRDA_OUTPUT_MODE")
        .env_remove("BIRDA_LATITUDE")
        .env_remove("BIRDA_LONGITUDE")
        .env_remove("HF_ENDPOINT")
        .timeout(COMMAND_TIMEOUT);
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.args(args);
    let output = cmd.output().expect("birda should run");
    Ran {
        status: output.status.code().expect("birda exits with a code"),
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn run(home: &Path, args: &[&str]) -> Ran {
    run_with(home, &[], args)
}

/// The JSON result of a run, without its volatile envelope fields.
fn result_of(ran: &Ran) -> Value {
    assert_eq!(ran.status, 0, "stderr: {}", ran.stderr);
    let mut doc: Value = serde_json::from_str(&ran.stdout).expect("stdout is one JSON document");
    let result = doc["payload"].take();
    assert!(!result.is_null(), "no payload in {doc}");
    result
}

fn seeded() -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    seed_registry(home.path(), "http://127.0.0.1:9");
    home
}

#[test]
fn test_models_regions_reports_the_pick_for_a_location() {
    let home = seeded();
    let home = home.path();

    // Tallinn: Baltics and Nordic tie at tier 50, Baltics is 311 km from its
    // centre and Nordic 538 km, the broad Western Palearctic follows.
    let tallinn = run(
        home,
        &[
            "--output-mode",
            "json",
            "models",
            "regions",
            "geo-test",
            "--lat",
            "59.44",
            "--lon",
            "24.75",
        ],
    );
    assert_eq!(
        result_of(&tallinn),
        json!({
            "result_type": "region_lookup",
            "model_id": "geo-test",
            "lat": 59.44,
            "lon": 24.75,
            "region": "baltics",
            "matches": [
                {"region": "baltics", "region_name": "Baltics", "tier": 50,
                 "classes": 479, "size_bytes": 102_547_724},
                {"region": "nordic", "region_name": "Nordic", "tier": 50,
                 "classes": 464, "size_bytes": 102_224_864},
                {"region": "western-palearctic", "region_name": "Western Palearctic",
                 "tier": 10, "classes": 1446, "size_bytes": 123_361_440},
            ],
        })
    );

    // Mid Atlantic: no tile, so the global model, and no `region` key at all.
    let sea = run(
        home,
        &[
            "--output-mode",
            "json",
            "models",
            "regions",
            "geo-test",
            "--lat",
            "30",
            "--lon",
            "-40",
        ],
    );
    assert_eq!(
        result_of(&sea),
        json!({
            "result_type": "region_lookup",
            "model_id": "geo-test",
            "lat": 30.0,
            "lon": -40.0,
            "matches": [],
        })
    );

    let human = run(
        home,
        &[
            "models", "regions", "geo-test", "--lat", "59.44", "--lon", "24.75",
        ],
    );
    assert_eq!(human.status, 0, "{}", human.stderr);
    assert_eq!(
        human.stdout,
        "Regions of Geo Test covering 59.44, 24.75:\n\
         \x20 baltics                   tier 50                  479 species   97.8 MB\n\
         \x20 nordic                    tier 50                  464 species   97.5 MB\n\
         \x20 western-palearctic        tier 10                 1446 species   117.6 MB\n\
         \n\
         Region for 59.44, 24.75: baltics (Baltics).\n\
         Also equally specific here: nordic.\n\
         \n\
         To install: birda models install geo-test --region baltics\n"
    );

    let human_sea = run(
        home,
        &[
            "models", "regions", "geo-test", "--lat", "30", "--lon", "-40",
        ],
    );
    assert_eq!(human_sea.status, 0, "{}", human_sea.stderr);
    assert_eq!(
        human_sea.stdout,
        "Regions of Geo Test covering 30, -40:\n\
         No regional model covers 30, -40.\n\
         \n\
         To install: birda models install geo-test --region global\n"
    );

    // A registry that publishes regions without boundaries cannot answer.
    let flat = run(
        home,
        &[
            "models",
            "regions",
            "flat-test",
            "--lat",
            "60",
            "--lon",
            "24",
        ],
    );
    assert_eq!(flat.status, 1);
    assert_eq!(flat.stdout, "");
    assert_eq!(
        flat.stderr,
        "error: model 'flat-test' publishes no region boundaries, so a region cannot be picked \
         from a location; choose one with --region <slug> (see 'birda models regions \
         flat-test')\n"
    );

    // Without a point the listing is what it was.
    let listing = run(home, &["models", "regions", "geo-test"]);
    assert!(
        listing
            .stdout
            .starts_with("Regional variants of Geo Test:\n"),
        "{}",
        listing.stdout
    );
}

#[test]
fn test_models_regions_picks_the_published_region_for_well_known_places() {
    // The bundled registry as shipped, in a fresh home, so this reads the real
    // BirdNET+ geometry.
    let home = tempfile::tempdir().unwrap();
    for (place, lat, lon, expected) in [
        ("Helsinki", "60.17", "24.94", Some("nordic")),
        ("Berlin", "52.52", "13.40", Some("central-europe")),
        ("Nagpur", "21.15", "79.09", Some("central-india")),
        ("Nairobi", "-1.29", "36.82", Some("east-africa")),
        ("Ithaca", "42.44", "-76.50", Some("north-america-east")),
        ("mid Atlantic", "30", "-40", None),
    ] {
        let ran = run(
            home.path(),
            &[
                "--output-mode",
                "json",
                "models",
                "regions",
                "birdnet-plus-v30",
                "--lat",
                lat,
                "--lon",
                lon,
            ],
        );
        let result = result_of(&ran);
        assert_eq!(result["region"].as_str(), expected, "{place}: {result}");
        let first = result["matches"][0]["region"].as_str();
        assert_eq!(first, expected, "{place}: the pick is the first match");
    }

    // Helsinki lists the broad region too, after the specific one.
    let helsinki = result_of(&run(
        home.path(),
        &[
            "--output-mode",
            "json",
            "models",
            "regions",
            "birdnet-plus-v30",
            "--lat",
            "60.17",
            "--lon",
            "24.94",
        ],
    ));
    let slugs: Vec<&str> = helsinki["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["region"].as_str().unwrap())
        .collect();
    assert_eq!(slugs, ["nordic", "western-palearctic"]);
}

#[test]
fn test_install_resolves_the_region_before_any_download() {
    let home = seeded();
    let home = home.path();

    // (a) `--region auto` with a point after `install`: the summary, the tie, the
    // action, then the selected variant, and the gate stops the install.
    let a = run(
        home,
        &[
            "models", "install", "geo-test", "--region", "auto", "--lat", "59.44", "--lon", "24.75",
        ],
    );
    assert_eq!(a.status, 1);
    assert_eq!(
        a.stdout,
        "Region for 59.44, 24.75: baltics (Baltics).\n\
         Also equally specific here: nordic.\n\
         Installing baltics.\n\
         \n\
         Selected variant fp32 (default, supported on every backend).\n\
         \x20 479 species, 97.8 MB\n\
         \n"
    );
    assert_eq!(a.stderr, GATE_REFUSAL);

    // (b) The configured location, with no flags.
    write_location(home, 60.17, 24.94);
    let b = run(home, &["models", "install", "geo-test", "--region", "auto"]);
    assert_eq!(b.status, 1);
    assert_eq!(
        b.stdout,
        "Region for 60.17, 24.94: nordic (Nordic).\n\
         Installing nordic.\n\
         \n\
         Selected variant fp32 (default, supported on every backend).\n\
         \x20 464 species, 97.5 MB\n\
         \n"
    );
    assert_eq!(b.stderr, GATE_REFUSAL);

    // (d) No `--region` and no terminal: the global model, exactly as before,
    // with a hint that regions exist. The configured location is not used.
    let d = run(home, &["models", "install", "geo-test"]);
    assert_eq!(d.status, 1);
    assert_eq!(
        d.stdout,
        "This model also publishes regional models: birda models regions geo-test, or \
         --region auto.\n\
         Selected variant fp32 (default, supported on every backend).\n\
         \x20 11725 species, 328.6 MB\n\
         \n"
    );
    assert_eq!(d.stderr, GATE_REFUSAL);

    // `--region global` pins the global model without the hint.
    let global = run(
        home,
        &["models", "install", "geo-test", "--region", "global"],
    );
    assert_eq!(global.status, 1);
    assert_eq!(
        global.stdout,
        "Selected variant fp32 (default, supported on every backend).\n\
         \x20 11725 species, 328.6 MB\n\
         \n"
    );

    // Under JSON nothing is printed before the gate's own error event.
    let json = run(
        home,
        &["--output-mode", "json", "models", "install", "geo-test"],
    );
    assert_eq!(json.status, 1);
    let event: Value = serde_json::from_str(&json.stdout).expect("one JSON document");
    assert_eq!(event["event"], "error", "{}", json.stdout);

    // (c) `--region auto` with no location anywhere.
    std::fs::remove_file(home.join("config.toml")).unwrap();
    let c = run(home, &["models", "install", "geo-test", "--region", "auto"]);
    assert_eq!(c.status, 1);
    assert_eq!(c.stdout, "");
    assert_eq!(
        c.stderr,
        "error: --region auto needs a location for 'geo-test': pass --lat and --lon after \
         'models install', or set defaults.latitude and defaults.longitude with 'birda config \
         set'\n"
    );
}

#[test]
fn test_install_refuses_a_location_it_would_not_use() {
    let home = seeded();
    let home = home.path();

    // `--region`, `--lat` and `--lon` never silently do nothing: the geomodel and
    // the bat models have no regions.
    for id in ["geomodel", "bat-anywhere"] {
        let ran = run(
            home,
            &[
                "models", "install", id, "--region", "auto", "--lat", "60", "--lon", "24",
            ],
        );
        assert_eq!(ran.status, 1, "{id}");
        assert_eq!(ran.stdout, "", "{id}");
        assert_eq!(
            ran.stderr,
            format!("error: model '{id}' has no regional variants\n"),
            "{id}"
        );
    }

    // A point only means something with `--region auto`.
    let named = run(
        home,
        &[
            "models", "install", "geo-test", "--region", "nordic", "--lat", "60", "--lon", "24",
        ],
    );
    assert_eq!(named.status, 1);
    assert_eq!(
        named.stderr,
        "error: --lat and --lon pick a region only with --region auto\n"
    );

    // `--lat` and `--lon` before `models` belong to analyze, on the command line
    // or in the environment, and never pick the install region, even beside a
    // configured location.
    write_location(home, 52.52, 13.4);
    let before_models = "error: --lat and --lon before 'models' (or \
        BIRDA_LATITUDE/BIRDA_LONGITUDE) are analyze options; pass --lat and --lon after 'models \
        install' to pick a region\n";
    let flags = run(
        home,
        &[
            "--lat", "60.17", "--lon", "24.94", "models", "install", "geo-test", "--region", "auto",
        ],
    );
    assert_eq!(flags.status, 1);
    assert_eq!(flags.stdout, "");
    assert_eq!(flags.stderr, before_models);
    let env = run_with(
        home,
        &[("BIRDA_LATITUDE", "60.17"), ("BIRDA_LONGITUDE", "24.94")],
        &["models", "install", "geo-test", "--region", "auto"],
    );
    assert_eq!(env.status, 1);
    assert_eq!(env.stdout, "");
    assert_eq!(env.stderr, before_models);

    // The same environment does not stop a point given where it belongs.
    let after = run_with(
        home,
        &[("BIRDA_LATITUDE", "60.17"), ("BIRDA_LONGITUDE", "24.94")],
        &[
            "models", "install", "geo-test", "--region", "auto", "--lat", "30", "--lon", "-40",
        ],
    );
    assert_eq!(after.status, 1);
    assert!(
        after
            .stdout
            .starts_with("No regional model covers 30, -40.\n"),
        "{}",
        after.stdout
    );
    assert_eq!(after.stderr, GATE_REFUSAL);
}

/// Answer `count` connections with a small `200 OK` body and return the base URL.
fn serve_downloads(count: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let deadline = Instant::now() + SERVER_DEADLINE;
        let mut served = 0;
        while served < count && Instant::now() < deadline {
            let Ok((mut stream, _)) = listener.accept() else {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            // Read the whole request head before replying, so closing the
            // socket does not reset the connection under the client.
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") && request.len() < 16_384 {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&chunk[..n]),
                }
            }
            let body = b"fixture";
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
            served += 1;
        }
    });
    base
}

#[test]
fn test_install_writes_the_region_it_reported() {
    // The region shown is the one installed (config key and JSON result), and a
    // plain non-interactive JSON install is the global model, as before regions.
    let home = tempfile::tempdir().unwrap();
    let base = serve_downloads(2 * DOWNLOADS_PER_INSTALL);
    seed_registry(home.path(), &base);

    let auto = run(
        home.path(),
        &[
            "--output-mode",
            "json",
            "models",
            "install",
            "geo-test",
            "--region",
            "auto",
            "--lat",
            "59.44",
            "--lon",
            "24.75",
            "--yes",
        ],
    );
    let result = result_of(&auto);
    assert_eq!(result["id"], "geo-test-baltics", "{result}");
    assert_eq!(result["region"], "baltics", "{result}");

    let plain = run(
        home.path(),
        &[
            "--output-mode",
            "json",
            "models",
            "install",
            "geo-test",
            "--yes",
        ],
    );
    let result = result_of(&plain);
    assert_eq!(result["id"], "geo-test", "{result}");
    assert!(result.get("region").is_none(), "{result}");

    let config = std::fs::read_to_string(home.path().join("config.toml")).unwrap();
    assert!(config.contains("[models.geo-test-baltics]"), "{config}");
    assert!(config.contains("[models.geo-test]"), "{config}");
}
