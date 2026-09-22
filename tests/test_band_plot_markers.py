"""BandPlot's VBM/CBM markers: drawn at the right (k, energy) position when a
BandGap is set, hidden on request, and cleared without error when the gap
disappears (calculation switch, metallic system, etc.).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vasp_analyzer.models.bands import BandData, BandGap, KPathLabel
from vasp_analyzer.parsers.procar import ProcarData
from vasp_analyzer.ui.pages.bands.band_plot import CBM_COLOR, VBM_COLOR, BandPlot
from vasp_analyzer.ui.pages.bands.sidebar import BandsSidebar

pytestmark = pytest.mark.usefixtures("qapp")


def _bands(labels: list[KPathLabel] | None = None) -> BandData:
    # Two k-points, two bands: k0 has the higher occupied state (VBM), k1 has
    # the lower unoccupied state (CBM) -- an indirect gap, like silicon.
    energies = np.array([[[-0.5, 3.0], [-2.0, 1.0]]])  # (spin=1, kpt=2, bands=2)
    return BandData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        weights=np.ones(2),
        energies=energies,
        occupations=np.zeros_like(energies),
        n_electrons=1.0,
        labels=labels or [],
    )


class _FakeGuiEvent:
    def globalPosition(self):
        from PySide6.QtCore import QPoint

        return SimpleNamespace(toPoint=lambda: QPoint(0, 0))


def _hover_event(axes, x, y, has_gui=True):
    return SimpleNamespace(
        inaxes=axes,
        xdata=x,
        ydata=y,
        guiEvent=_FakeGuiEvent() if has_gui else None,
    )


def _gap() -> BandGap:
    bands = _bands()
    result = bands.band_gap(e_fermi=0.0)
    assert result is not None
    return result


def _marker_colors(plot: BandPlot) -> list[str]:
    from matplotlib.colors import to_hex

    return [
        to_hex(collection.get_facecolor()[0])
        for collection in plot.axes.collections
        if collection.get_offsets().shape[0] == 1
    ]


def test_markers_appear_at_the_vbm_and_cbm_colors_when_a_gap_is_set():
    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_band_gap(_gap())

    colors = _marker_colors(plot)
    assert set(colors) == {VBM_COLOR, CBM_COLOR}


def test_markers_hidden_when_visibility_toggled_off():
    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_band_gap(_gap())
    plot.set_band_gap_visible(False)

    assert _marker_colors(plot) == []


def test_no_markers_and_no_error_when_gap_is_none():
    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_band_gap(None)  # e.g. a metallic calculation

    assert _marker_colors(plot) == []


def test_switching_calculations_clears_stale_markers():
    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_band_gap(_gap())
    assert _marker_colors(plot) != []

    plot.set_bands(None)
    assert len(plot.axes.collections) == 0


def test_sidebar_checkbox_is_checked_by_default_and_emits_on_toggle(qtbot):
    sidebar = BandsSidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.show_band_gap.isChecked()

    with qtbot.waitSignal(sidebar.bandGapMarkersToggled, timeout=1000) as blocker:
        sidebar.show_band_gap.setChecked(False)
    assert blocker.args == [False]


def test_hover_info_checkbox_is_checked_by_default_and_emits_on_toggle(qtbot):
    sidebar = BandsSidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.show_hover_info.isChecked()

    with qtbot.waitSignal(sidebar.hoverInfoToggled, timeout=1000) as blocker:
        sidebar.show_hover_info.setChecked(False)
    assert blocker.args == [False]


def test_set_hover_enabled_false_suppresses_the_tooltip():
    from PySide6.QtWidgets import QToolTip

    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_hover_enabled(False)

    plot._on_hover(_hover_event(plot.axes, x=1.0, y=1.0))
    assert QToolTip.text() == ""


def test_set_hover_enabled_true_restores_the_tooltip():
    from PySide6.QtWidgets import QToolTip

    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_hover_enabled(False)
    plot.set_hover_enabled(True)

    plot._on_hover(_hover_event(plot.axes, x=1.0, y=1.0))
    assert "k-point 1" in QToolTip.text()


# --- hover tooltip -----------------------------------------------------------


def test_hover_shows_tooltip_with_kpoint_band_and_energy():
    from PySide6.QtWidgets import QToolTip

    plot = BandPlot()
    plot.set_bands(_bands())

    # k-point 1, band index 1: energies[0, 1] == [-2.0, 1.0] -> band 2 is 1.0 eV.
    plot._on_hover(_hover_event(plot.axes, x=1.0, y=1.0))

    text = QToolTip.text()
    assert "k-point 1" in text
    assert "Band 2" in text
    assert "1.000 eV" in text


def test_hover_ignores_points_far_from_any_band(monkeypatch):
    # QToolTip's own visibility/text() don't reflect show/hide calls under
    # the offscreen platform used for tests (no real window manager), so the
    # call itself -- not the tooltip widget's resulting state -- is what's
    # observable here.
    import vasp_analyzer.ui.pages.bands.band_plot as band_plot_module

    hidden = []
    monkeypatch.setattr(band_plot_module.QToolTip, "hideText", lambda: hidden.append(True))

    plot = BandPlot()
    plot.set_bands(_bands())

    # y=8 at k-point 1 (bands at -2.0, 1.0) is far outside the 0.03*range
    # tolerance -- empty space between bands should stay quiet.
    plot._on_hover(_hover_event(plot.axes, x=1.0, y=8.0))
    assert hidden == [True]


def test_hover_outside_the_axes_hides_the_tooltip(monkeypatch):
    import vasp_analyzer.ui.pages.bands.band_plot as band_plot_module

    hidden = []
    monkeypatch.setattr(band_plot_module.QToolTip, "hideText", lambda: hidden.append(True))

    plot = BandPlot()
    plot.set_bands(_bands())
    plot._on_hover(_hover_event(None, x=1.0, y=1.0))
    assert hidden == [True]


def test_kpoint_label_uses_the_high_symmetry_name_when_present():
    plot = BandPlot()
    bands = _bands(labels=[KPathLabel(index=1, position=1.0, label="Γ")])
    plot.set_bands(bands)

    assert plot._k_point_label(0, bands) == "k-point 0"
    assert plot._k_point_label(1, bands) == "k-point 1 (Γ)"


# --- orbital breakdown --------------------------------------------------------


def _procar_two_ion() -> ProcarData:
    orbitals = ["s", "p"]
    # (n_spin=1, n_kpoints=2, n_bands=2, n_ions=2, n_orbitals=2)
    ion_weights = np.zeros((1, 2, 2, 2, 2))
    # spin0, k0, band0: ion0 (Si) mostly p, ion1 (O) mostly s.
    ion_weights[0, 0, 0] = [[0.05, 0.65], [0.25, 0.05]]
    return ProcarData(orbitals=orbitals, ion_weights=ion_weights)


def test_orbital_breakdown_groups_by_element_and_sorts_by_weight():
    plot = BandPlot()
    plot.set_bands(_bands())
    plot.set_orbital_source(_procar_two_ion(), ["Si", "O"])

    breakdown = plot._orbital_breakdown(spin=0, k_index=0, band_index=0)

    assert breakdown == ["Si p: 65%", "O s: 25%", "O p: 5%", "Si s: 5%"]


def test_orbital_breakdown_empty_without_a_procar_source():
    plot = BandPlot()
    plot.set_bands(_bands())
    assert plot._orbital_breakdown(0, 0, 0) == []


def test_orbital_breakdown_empty_when_procar_dimensions_mismatch_bands():
    plot = BandPlot()
    plot.set_bands(_bands())
    mismatched = ProcarData(orbitals=["s"], ion_weights=np.zeros((1, 5, 5, 2, 1)))
    plot.set_orbital_source(mismatched, ["Si", "O"])

    assert plot._orbital_breakdown(0, 0, 0) == []
