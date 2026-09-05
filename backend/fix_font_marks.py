import os
import fontforge

# Exact path to your font file based on your VS Code project structure
FONT_PATH = os.path.abspath(
    "../frontend/assets/fonts/baybayin_custom.ttf"
)

def fix_baybayin_marks():
    if not os.path.exists(FONT_PATH):
        print(f"Error: Could not find font file at {FONT_PATH}")
        return

    print(f"Opening font: {FONT_PATH}")
    font = fontforge.open(FONT_PATH)

    # Shift value: -400 moves the mark into negative space over the previous base letter
    target_x = -1000  

    marks = ["uni1712", "uni1713", "uni1714", "uni1715", "uni1716"]

    for mark_name in marks:
        if mark_name in font:
            glyph = font[mark_name]
            
            # Calculate current visual center of the glyph
            bbox = glyph.boundingbox()
            current_center_x = (bbox[0] + bbox[2]) / 2.0
            
            # Calculate exact distance to move left
            shift_x = target_x - current_center_x
            
            # Move vector points and force advance width to 0
            glyph.transform((1, 0, 0, 1, shift_x, 0))
            glyph.width = 0
            print(f"Fixed {mark_name}: Shifted {shift_x} units left, width set to 0.")

    # Overwrite the font file in place
    font.generate(FONT_PATH)
    print("Successfully updated baybayin_custom.ttf!")

if __name__ == "__main__":
    fix_baybayin_marks()