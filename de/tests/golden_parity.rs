use ariacompute_core::packing::Record;
use ariacompute_de::score::score_record_logits;
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
fn encoder_choice_golden() {
    let fx = fixture("encoder_choice.json");
    let rec = Record::from_value(&fx).unwrap();
    let logits: Vec<f32> = fx["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let ans = score_record_logits(&rec, &logits, 1.0).unwrap();
    assert_eq!(ans["choice"], fx["expect_choice"]);
}

#[test]
fn encoder_noul_golden() {
    let fx = fixture("encoder_noul.json");
    let rec = Record::from_value(&fx).unwrap();
    let logits: Vec<f32> = fx["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let ans = score_record_logits(&rec, &logits, 1.0).unwrap();
    let noul = ans["noul"].as_f64().unwrap();
    assert!(noul > fx["expect_noul_gt"].as_f64().unwrap());
}
