import numpy as np

from cfplot import contour


def test_con_delegates_to_legacy(monkeypatch):
    called = {}

    def fake_legacy_con(*, f=None, x=None, y=None, **kwargs):
        called["f"] = f
        called["x"] = x
        called["y"] = y
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("cfplot.cfplot._legacy_con", fake_legacy_con)

    out = contour.con(f="field", x="xvals", y="yvals", lines=False)

    assert out == "ok"
    assert called["f"] == "field"
    assert called["x"] == "xvals"
    assert called["y"] == "yvals"
    assert called["kwargs"]["lines"] is False


def test_colour_scale_label_skip():
    cs = contour.ColourScale(plotvars=object())

    labels = cs.colorbar_labels(
        levels=np.array([0, 1, 2, 3]),
        orientation="horizontal",
        n_columns=1,
        label_skip=2,
        custom_labels=None,
    )

    assert labels == ["0", "", "2", ""]