"""SliderRow layout: the inline variant's min/max captions must track the
slider's own width, not the whole row (value box included)."""

from __future__ import annotations

from vasp_analyzer.ui.widgets.slider_row import SliderRow


def test_inline_captions_share_a_layout_with_the_slider_not_the_value_box(qtbot):
    """Regression: the captions row used to be appended full-width beneath
    the whole inline row (slider + value box + unit), so the max label
    landed under the value box's corner instead of the slider's right edge —
    exactly what "the 10 needs to be under the bar" reported. The min/max
    labels must now live in the same column as the slider, separate from the
    value box."""
    row = SliderRow("", 0.0, 10.0, 5.0, 1.0, "", inline=True)
    qtbot.addWidget(row)

    def find_containing(layout, widget):
        if layout.indexOf(widget) != -1:
            return layout
        for i in range(layout.count()):
            sub = layout.itemAt(i).layout()
            if sub is not None:
                found = find_containing(sub, widget)
                if found is not None:
                    return found
        return None

    slider_column = find_containing(row.layout(), row._slider)
    assert slider_column is not None

    # The slider and both captions sit in the same column together...
    found = False
    for i in range(slider_column.count()):
        sub = slider_column.itemAt(i).layout()
        if sub is not None and sub.indexOf(row._max_label) != -1:
            found = True
    assert found, "max label must share the slider's own column"

    # The value box must NOT be in that column — it sits beside it.
    assert slider_column.indexOf(row._field) == -1


def test_non_inline_captions_still_span_the_full_slider_width(qtbot):
    """The non-inline layout was already correct — the slider has the whole
    row to itself, so a plain full-width captions row beneath it already
    lines up. Must not regress."""
    row = SliderRow("Title", 0.0, 10.0, 5.0, 1.0, "eV", inline=False)
    qtbot.addWidget(row)

    outer = row.layout()
    found = False
    for i in range(outer.count()):
        sub = outer.itemAt(i).layout()
        if sub is not None and sub.indexOf(row._max_label) != -1:
            found = True
    assert found


def test_range_captions_show_the_configured_min_and_max(qtbot):
    row = SliderRow("", 0.0, 1000.0, 6.61, 1.0, "", inline=True)
    qtbot.addWidget(row)
    assert row._min_label.text() == "0"
    assert row._max_label.text() == "1000"
