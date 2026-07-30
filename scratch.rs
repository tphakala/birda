use std::path::PathBuf;
fn check(keeping: &[PathBuf], still_referenced: Vec<&PathBuf>, p: PathBuf) -> bool {
    let arr = [p.clone()];
    arr.into_iter().filter(|path| !keeping.contains(path)).filter(|path| !still_referenced.contains(&path)).count() == 0
}
