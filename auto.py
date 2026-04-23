# Запусти это ОТДЕЛЬНО перед train_uz.py

from pathlib import Path

CHARS = ['0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'L','M','N','O','P','Q','R','S','T','U',
         'V','W','X','Y','Z']
CHARS_DICT = {c: i for i, c in enumerate(CHARS)}
BLANK_IDX  = len(CHARS)  # 35

for split in ['train', 'val']:
    folder = Path(fr'C:\Users\user\Pictures\ANPR\dataset\final\{split}')
    print(f"\n=== {split} ===")
    bad = []
    for fpath in sorted(folder.glob("*.jpg")):
        label = fpath.stem.split('_')[0]
        for c in label:
            if c not in CHARS_DICT:
                bad.append((fpath.name, label, c))
    
    if bad:
        print(f"❌ Найдено проблемных: {len(bad)}")
        for fname, label, bad_char in bad:
            print(f"   {fname} → label={label} → плохой символ='{bad_char}' (ord={ord(bad_char)})")
    else:
        print(f"✅ Все символы корректны")