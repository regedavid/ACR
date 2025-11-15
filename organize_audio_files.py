import os
import shutil
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import re
import unicodedata
import librosa
import soundfile as sf
import difflib

# ---------------------------------------------------------
# MAP ALBUM METADATA TO MIRDATA FOLDER NAMES
# ---------------------------------------------------------
ALBUM_MAP = {
    # 1963
    "Please Please Me": "01_-_Please_Please_Me",
    "Please Please Me - Remastered": "01_-_Please_Please_Me",
    "Please Please Me (Remastered)": "01_-_Please_Please_Me",
    "Please Please Me (2009 Remaster)": "01_-_Please_Please_Me",

    # 1963
    "With The Beatles": "02_-_With_the_Beatles",
    "With The Beatles (Remastered)": "02_-_With_the_Beatles",
    "With the Beatles (2009 Remaster)": "02_-_With_the_Beatles",

    # 1964
    "A Hard Day's Night": "03_-_A_Hard_Day's_Night",
    "A Hard Day's Night (Remastered)": "03_-_A_Hard_Day's_Night",
    "A Hard Day's Night (2009 Remaster)": "03_-_A_Hard_Day's_Night",

    # 1964
    "Beatles for Sale": "04_-_Beatles_for_Sale",
    "Beatles For Sale (Remastered)": "04_-_Beatles_for_Sale",
    "Beatles for Sale (2009 Remaster)": "04_-_Beatles_for_Sale",

    # 1965
    "Help!": "05_-_Help!",
    "Help! (Remastered)": "05_-_Help!",
    "Help! (Remastered 2009)": "05_-_Help!",
    "Help! (2009 Remaster)": "05_-_Help!",

    # 1965
    "Rubber Soul": "06_-_Rubber_Soul",
    "Rubber Soul (Remastered)": "06_-_Rubber_Soul",
    "Rubber Soul (2009 Remaster)": "06_-_Rubber_Soul",

    # 1966
    "Revolver": "07_-_Revolver",
    "Revolver (Remastered)": "07_-_Revolver",
    "Revolver (2009 Remaster)": "07_-_Revolver",

    # 1967
    "Sgt. Pepper's Lonely Hearts Club Band": "08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band",
    "Sgt. Pepper's Lonely Hearts Club Band (Remastered)": "08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band",
    "Sgt. Pepper's Lonely Hearts Club Band (2009 Remaster)": "08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band",

    # 1967
    "Magical Mystery Tour": "09_-_Magical_Mystery_Tour",
    "Magical Mystery Tour (Remastered)": "09_-_Magical_Mystery_Tour",
    "Magical Mystery Tour (2009 Remaster)": "09_-_Magical_Mystery_Tour",

    # 1968
    "The Beatles": "10_-_The_Beatles",
    "The Beatles (Remastered)": "10_-_The_Beatles",
    "The Beatles (2009 Remaster)": "10_-_The_Beatles",
    "The Beatles (The White Album) (Remastered)": "10_-_The_Beatles",

    # 1969
    "Yellow Submarine": "15_-_Yellow_Submarine",
    "Yellow Submarine (Remastered)": "15_-_Yellow_Submarine",
    "Yellow Submarine (2009 Remaster)": "15_-_Yellow_Submarine",

    # 1969
    "Abbey Road": "11_-_Abbey_Road",
    "Abbey Road (Remastered)": "11_-_Abbey_Road",
    "Abbey Road (2009 Remaster)": "11_-_Abbey_Road",

    # 1970
    "Let It Be": "12_-_Let_It_Be",
    "Let It Be (Remastered)": "12_-_Let_It_Be",
    "Let It Be (2009 Remaster)": "12_-_Let_It_Be",

    # Singles collection
    "Past Masters (Remastered)": "14_-_Past_Masters",
    "Past Masters, Vols. 1 & 2 (2009 Remaster)": "14_-_Past_Masters",
    "Past Masters, Vol. 1 & 2 (Remastered)": "14_-_Past_Masters",
}


# ---------------------------------------------------------
# CLEAN/SAFE STRING FOR FILENAMES
# ---------------------------------------------------------
def clean_name(name: str) -> str:
    # Normalize unicode
    name = unicodedata.normalize("NFKC", name)

    # Remove anything that indicates a remaster
    name = re.sub(r"\s*\(.*?Remaster.*?\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-\s*Remaster.*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"Remaster(ed)?(\s*\d{4})?", "", name, flags=re.IGNORECASE)

    # Remove leftover double spaces
    name = re.sub(r"\s{2,}", " ", name).strip()

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove unsafe characters for filenames
    name = re.sub(r"[^A-Za-z0-9_\-']", "", name)

    return name

# ---------------------------------------------------------
# MAIN ORGANIZER
# ---------------------------------------------------------
def organize_mp3_folder(source_folder, output_folder="Beatles_mp3/audio"):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(source_folder):
        if not fname.lower().endswith(".mp3"):
            continue

        path = os.path.join(source_folder, fname)
        audio = MP3(path, ID3=ID3)

        if "TALB" not in audio or "TIT2" not in audio or "TRCK" not in audio:
            print(f"Skipping (missing metadata) → {fname}")
            continue

        album = str(audio["TALB"])
        title = str(audio["TIT2"])
        track_str = str(audio["TRCK"])

        # Track number can be "1" or "1/14"
        track_num = int(track_str.split("/")[0])

        # Match album to mirdata folder
        if album not in ALBUM_MAP:
            print(f"Unknown album '{album}' for file: {fname}")
            continue

        album_folder = os.path.join(output_folder, ALBUM_MAP[album])
        os.makedirs(album_folder, exist_ok=True)

        # Create destination filename
        clean_title = clean_name(title)
        dest_name = f"{track_num:02d}_-_{clean_title}.mp3"
        dest_path = os.path.join(album_folder, dest_name)

        print(f"{fname} → {dest_path}")
        shutil.copy2(path, dest_path)

    print("\n✔ Done organizing Beatles MP3 files for mirdata.")


def mp3_to_wav(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_folder):
        for fname in files:
            if not fname.lower().endswith(".mp3"):
                continue

            mp3_path = os.path.join(root, fname)
        
            # Load audio
            y, sr = librosa.load(mp3_path, sr=None)  # sr=None preserves original sampling rate
            
            # Convert filename to .wav
            root_rel = os.path.relpath(root, source_folder)
            target_folder_full = os.path.join(target_folder, root_rel)
            os.makedirs(target_folder_full, exist_ok=True)
            wav_name = os.path.splitext(fname)[0] + ".wav"
            wav_path = os.path.join(target_folder_full, wav_name)
            
            # Save as WAV
            sf.write(wav_path, y, sr)
            
            print(f"{mp3_path} → {wav_path}")


def normalize_name(name):
    """
    Lowercase, remove extension, replace spaces with underscores,
    remove unsafe characters but keep apostrophes and underscores.
    """
    name = os.path.splitext(name)[0]  # remove extension
    name = name.lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_']", "", name)
    return name

def get_all_files(folder, extension):
    """Recursively get all files with the given extension."""
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(extension.lower()):
                files.append(os.path.join(root, f))
    return files

def match_wav_to_lab(lab_folder, wav_folder, rename=False):
    # Get all files recursively
    lab_files = get_all_files(lab_folder, ".lab")
    wav_files = get_all_files(wav_folder, ".wav")

    wav_norm = {f: normalize_name(os.path.basename(f)) for f in wav_files}
    mapping = {}

    for lab_path in lab_files:
        lab_name = os.path.basename(lab_path)  # keep exact lab filename
        lab_key = normalize_name(lab_name)     # for matching purposes only

        # Find closest match in wav files
        matches = difflib.get_close_matches(lab_key, wav_norm.values(), n=1, cutoff=0.6)
        if matches:
            best_match_key = matches[0]
            wav_path = next(f for f, norm in wav_norm.items() if norm == best_match_key)
            mapping[lab_path] = wav_path

            if rename:
                # Rename wav to exactly match lab filename
                new_wav_path = os.path.join(os.path.dirname(wav_path),
                                            lab_name.replace(".lab", ".wav"))
                os.rename(wav_path, new_wav_path)
                mapping[lab_path] = new_wav_path
        else:
            mapping[lab_path] = None

    return mapping
    
# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    #organize_mp3_folder(source_folder="/mnt/c/Users/Dwawid/Downloads/the-beatles-1964-beatles-for-sale", output_folder="Beatles_og/audio")
    mp3_to_wav("Beatles_og/audio", "Beatles_og_wav/audio")
    
    # lab_folder = "/home/dwawid/mir_datasets/beatles/annotations/chordlab"
    # wav_folder = "/home/dwawid/mir_datasets/beatles/audio"
    # mapping = match_wav_to_lab(lab_folder, wav_folder, rename=True)

    # for lab, wav in mapping.items():
    #     if wav is None:
    #         print(f"No match for {lab}")
    #     else:
    #         print(f"{lab} → {wav}")
