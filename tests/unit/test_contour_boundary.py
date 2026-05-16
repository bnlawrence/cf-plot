import numpy as np
import pytest

from cfplot import contour


def test_con_delegates_to_legacy(monkeypatch):
    monkeypatch.setattr("cfplot.contour._can_use_new_xy_path", lambda f, kwargs: False)

    with pytest.raises(NotImplementedError, match="not implemented"):
        contour.con(f=np.array([[1.0, 2.0], [3.0, 4.0]]), lines=False)


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


def test_con_uses_new_path_when_available(monkeypatch):
    def fake_legacy_con(*, f=None, x=None, y=None, **kwargs):
        raise AssertionError("legacy path should not be called")

    monkeypatch.setattr("cfplot.cfplot._legacy_con", fake_legacy_con)
    monkeypatch.setattr("cfplot.contour._can_use_new_xy_path", lambda f, kwargs: True)
    monkeypatch.setattr(
        "cfplot.contour._render_with_new_xy",
        lambda f, x, y, kwargs: True,
    )

    out = contour.con(f=np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert out is None


def test_con_falls_back_when_new_path_declines(monkeypatch):
    monkeypatch.setattr("cfplot.contour._can_use_new_xy_path", lambda f, kwargs: True)
    monkeypatch.setattr(
        "cfplot.contour._render_with_new_xy",
        lambda f, x, y, kwargs: False,
    )

    with pytest.raises(NotImplementedError, match="not implemented"):
        contour.con(f=np.array([[1.0, 2.0], [3.0, 4.0]]), lines=False)


def test_finalize_non_session_plot_calls_gclose(monkeypatch):
    calls: list[bool] = []

    monkeypatch.setattr(contour.plotvars, "_contour_session_open", False)
    monkeypatch.setattr(contour, "gclose", lambda view=True: calls.append(view))

    contour._finalize_non_session_plot()

    assert calls == [True]


def test_finalize_non_session_plot_skips_when_session_open(monkeypatch):
    calls: list[bool] = []

    monkeypatch.setattr(contour.plotvars, "_contour_session_open", True)
    monkeypatch.setattr(contour, "gclose", lambda view=True: calls.append(view))

    contour._finalize_non_session_plot()

    assert calls == []