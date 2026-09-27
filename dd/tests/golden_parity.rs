use ariacompute_core::packing::Record;
use ariacompute_dd::DecoderScorer;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixture(name: &str) -> Value {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../tests/fixtures");
    path.push(name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn decoder_choice_golden() {
    let fx = fixture("decoder_choice.json");
    let rec = Record::from_value(&fx).unwrap();
    let scorer = DecoderScorer::open(None::<&str>).unwrap();
    let scored = scorer
        .score_from_semif_out(&rec, &fx["semif_out"])
        .unwrap();
    assert_eq!(scored["systemone"]["choice"], fx["expect_choice"]);
}
