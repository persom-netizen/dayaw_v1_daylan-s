import re

class TagalogToBaybayin:
    def __init__(self):
        self.base_map = {
            # Vowels
            # Standalone vowels. Bare I and U share a base glyph with E and O,
            # so the lone letters map to dedicated glyphs in baybayin_custom.ttf
            # (E010 = the I/E shape + a short vertical line centred above;
            #  E011 = the U/O shape + a short vertical tick at the bottom-right).
            # This keeps a typed "i" / "u" from round-tripping through the BTL
            # OCR as "e" / "o". Consonant+i/u are unaffected (they use the
            # kudlit marks below).
            'a': 'ᜀ', 'e': 'ᜁ', 'i': '', 'o': 'ᜂ', 'u': '',

            # Consonants with inherent A vowel
            'ba': 'ᜊ', 'ka': 'ᜃ', 'da': 'ᜇ',
            'ra': 'ᜍ',
            'ga': 'ᜄ', 'ha': 'ᜑ', 'la': 'ᜎ',
            'ma': 'ᜋ', 'na': 'ᜈ', 'nga': 'ᜅ',
            'pa': 'ᜉ', 'sa': 'ᜐ', 'ta': 'ᜆ',
            'wa': 'ᜏ', 'ya': 'ᜌ'
        }

        # ==============================================================
        # 4 SEPARATE KUDLIT MARKS:
        # E → dash above  U+1732 (edit existing in FontForge)
        # I → dot above   U+E001 (new PUA slot in FontForge)
        # U → dash below  U+1733 (edit existing in FontForge)
        # O → dot below   U+E002 (new PUA slot in FontForge)
        # Cross/x → virama U+1714 (already done)
        # ==============================================================
        self.kudlit_e = '\u1715'   # dash above — E
        self.kudlit_i = '\u1712'   # dot above  — I
        self.kudlit_u = '\u1716'   # dash below — U
        self.kudlit_o = '\u1713'   # dot below  — O
        self.virama       = '\u1714'   # cross/x — single consonant
        self.danda        = '᜵'
        self.double_danda = '᜶'

    def translate(self, text):
        if not text:
            return "", 0

        original_text = text.lower().strip()
        confidence    = 100.0

        non_native = re.findall(r'[cfjqzvx]', original_text)
        if non_native:
            confidence -= (len(non_native) * 15)

        text = original_text.replace('ng', 'NG')

        if text == "mga":
            return "ᜋᜄ", 100.0

        pattern = r'(NG[aeiou]|(?:[bkdrghlmnpstwry])?[aeiou])|(NG|[bkdrghlmnpstwry])|([aeiou])|(\s+)|(\.|\,)'
        tokens  = re.findall(pattern, text)
        result  = []

        for cv, c, v, space, punct in tokens:

            if space:
                result.append(space)
                continue

            if punct:
                if punct == '.':
                    result.append(self.double_danda)
                elif punct == ',':
                    result.append(self.danda)
                continue

            vowel_token = v if v else (cv if cv in 'aeiou' else None)
            if vowel_token:
                result.append(self.base_map.get(vowel_token, ''))
                continue

            elif cv:
                vowel_part = cv[-1]
                cons_part  = cv[:-1]

                if cons_part == 'r':
                    key = 'ra'
                elif cons_part == 'NG':
                    key = 'nga'
                else:
                    key = cons_part + 'a'

                base = self.base_map.get(key, '')

                # ==================================================
                # 4 SEPARATE KUDLIT MARKS
                # E → dash above  ✅
                # I → dot above   ✅
                # O → dot below   ✅
                # U → dash below  ✅
                # ==================================================
                if vowel_part == 'e':
                    result.append(base + self.kudlit_e)   # dash above
                elif vowel_part == 'i':
                    result.append(base + self.kudlit_i)   # dot above
                elif vowel_part == 'o':
                    result.append(base + self.kudlit_o)   # dot below
                elif vowel_part == 'u':
                    result.append(base + self.kudlit_u)   # dash below
                else:
                    result.append(base)                    # a — no kudlit

            elif c:
                if c == 'r':
                    key = 'ra'
                elif c == 'NG':
                    key = 'nga'
                else:
                    key = c + 'a'

                base = self.base_map.get(key, '')
                if base:
                    result.append(base + self.virama)      # cross/x

        return "".join(result), max(0, confidence)