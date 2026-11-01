from nids.datasets import detect_label_column, identifier_columns


def test_cicids_rejects_wsn_label_name():
    try:
        detect_label_column(["Flow Duration", "Attack type"], "cicids2018")
        assert False, "should have rejected WSN-DS label on CICIDS2018"
    except ValueError as exc:
        assert "Label" in str(exc)


def test_wsn_accepts_attack_type():
    assert detect_label_column(["id", "Attack type"], "wsn_ds") == "Attack type"


def test_identifier_drop_lists():
    wsn = identifier_columns(["id", "who CH", "Rank", "Attack type"], "wsn_ds")
    assert "id" in wsn
    assert "who CH" in wsn
    cic = identifier_columns(
        ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Dst Port", "Label"],
        "cicids2018",
    )
    assert set(cic) == {"Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"}
