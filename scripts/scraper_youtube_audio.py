#!/usr/bin/env python3
"""
Scraper Audio YouTube - Lagu Daerah Indonesia (Professional Studio Recording Curation)
Dirancang khusus untuk proyek Deep Learning (Semester 5).

Fitur Utama:
1. Menyaring video secara ketat: Anti-lagu anak/kartun, anti-remix/jedag-jedug, anti-SFX,
   anti-konten bukan lagu (berita, cerita legenda, medley/kumpulan, rekaman siswa).
2. Judul video WAJIB mirip dengan nama lagu (atau nama lainnya) -> mencegah salah unduh.
3. Satu video YouTube hanya boleh dipakai untuk SATU lagu -> mencegah duplikat/label bertentangan.
4. Memilih kandidat dengan skor tertinggi (kecocokan judul + label resmi/Topic), bukan yang pertama lolos.
5. Memprioritaskan rilisan studio resmi (YouTube Topic, GNP Music, Official Audio).
6. Mendukung konversi audio ke M4A/MP3/WAV/FLAC (menggunakan ffmpeg).
7. Manifest audio_manifest.csv di-update per id lagu (tidak menumpuk baris ganda) + backup otomatis.
8. Mode scrape ulang: --ids / --dari-cek-manual + --force.
"""

import argparse
import csv
import difflib
import os
import re
import shutil
import sys
import time
import unicodedata
from datetime import datetime
import yt_dlp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lokasi default ffmpeg di environment conda deeplearning
DEFAULT_FFMPEG_PATHS = [
    r"C:\Users\chris\anaconda3\envs\deeplearning\Library\bin\ffmpeg.exe",
    r"C:\Users\chris\anaconda3\Library\bin\ffmpeg.exe",
]

# Pola blacklist: eliminasi mutlak untuk video anak, kartun, senam, remix, dll.
BLACKLIST_REGEX = re.compile(
    r'\b(anak|anak2|anak-anak|balita|paud|tk|kartun|animasi|animation|'
    r'cocomelon|didi|senam|tari|tarian|gerak|lomba|festival|'
    r'remix|dj|jedag|jedug|bass\s*boost|koplo|slowed|reverb|speed\s*up|'
    r'karaoke|minus\s*one|lirik\s*tanpa\s*vokal|instrumental\s*karaoke|'
    r'vlog|drama|short|shorts|tiktok|reels|parodi|meme|'
    # Konten bukan lagu tunggal (temuan EDA: klip berita, cerita legenda, medley, rekaman siswa)
    r'berita|news|liputan|cerita|legenda|dongeng|asal\s*usul|podcast|tutorial|'
    r'medley|kumpulan|nonstop|non\s*stop|full\s*album|compilation|kompilasi|'
    r'siswa|siswi|sd|smp|sma|smk|mts)\b',
    re.IGNORECASE
)

# Label rekaman atau penerbit musik tradisional resmi
OFFICIAL_LABELS = [
    "gema nada pertiwi", "gnp music", "gnp", "irama tara",
    "musica studios", "aquarius musikindo", "swaragita",
    "dasa studio", "virgo putra", "surya eka"
]

# Ambang minimal kecocokan judul video dengan nama lagu (0-1)
MIN_KEMIRIPAN_JUDUL = 0.6

# Kata umum yang diabaikan saat mencocokkan judul
KATA_UMUM = {"lagu", "daerah", "official", "audio", "video", "music", "musik", "lirik", "lyrics", "lyric", "the"}


def resolve_ffmpeg() -> str | None:
    """Mencari path binary ffmpeg yang valid."""
    for p in DEFAULT_FFMPEG_PATHS:
        if os.path.exists(p):
            return p
    # Fallback ke PATH sistem
    return shutil.which("ffmpeg")


def sanitize_filename(name: str) -> str:
    """Membersihkan karakter ilegal untuk nama berkas sistem operasi."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
    cleaned = re.sub(r'\s+', "_", cleaned.strip())
    return cleaned


def resolve_project_path(path: str) -> str:
    """Selesaikan path relatif terhadap lokasi proyek, bukan direktori terminal."""
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


def manifest_audio_path(path: str) -> str:
    """Simpan path relatif ke proyek jika audio berada di dalam folder proyek."""
    absolute_path = os.path.abspath(path)
    try:
        relative_path = os.path.relpath(absolute_path, BASE_DIR)
    except ValueError:  # Mis. audio disimpan pada drive lain di Windows.
        return absolute_path
    if relative_path != os.pardir and not relative_path.startswith(os.pardir + os.sep):
        return relative_path.replace(os.sep, "/")
    return absolute_path


def _token(teks: str) -> list[str]:
    """Normalisasi teks (huruf kecil, tanpa aksen/tanda baca) lalu pecah jadi kata."""
    teks = unicodedata.normalize("NFKD", teks).encode("ascii", "ignore").decode()
    teks = re.sub(r"[^a-z0-9 ]", " ", teks.lower().replace("'", ""))
    return [t for t in teks.split() if t not in KATA_UMUM]


def kemiripan_judul(nama_lagu: str, judul: str) -> float:
    """
    Skor 0-1: proporsi kata nama lagu yang muncul di judul video.
    Toleran terhadap beda ejaan kecil (mis. 'Bungong' vs 'Bungung') dan tanda hubung.
    """
    kata_lagu, kata_judul = _token(nama_lagu), _token(judul)
    if not kata_lagu or not kata_judul:
        return 0.0
    # Nama lagu ditulis tanpa spasi/tanda hubung (mis. 'AmparAmpar Pisang')
    if "".join(kata_lagu) in "".join(kata_judul):
        return 1.0
    cocok = 0
    for k in kata_lagu:
        terbaik = max(difflib.SequenceMatcher(None, k, j).ratio() for j in kata_judul)
        cocok += terbaik >= 0.8
    return cocok / len(kata_lagu)


def daftar_nama(song: dict) -> list[str]:
    """Nama utama + nama lain lagu (kolom nama_lain, dipisah ; / ,)."""
    nama = [song.get("nama_lagu", "")]
    nama += [n.strip() for n in re.split(r"[;/,]", song.get("nama_lain", "") or "") if n.strip()]
    return [n for n in nama if n]


def validate_candidate(
    entry: dict,
    song_names: list[str] | str = "",
    province: str = "",
    blocked_ids: set[str] | None = None,
) -> tuple[bool, str, float]:
    """
    Memvalidasi apakah kandidat video YouTube layak diunduh.
    Memeriksa durasi, blacklist, kecocokan judul, dan duplikasi video.
    Mengembalikan (is_valid, alasan, kemiripan_judul).
    """
    title = entry.get("title", "") or ""
    channel = entry.get("channel", "") or entry.get("uploader", "") or ""
    desc = entry.get("description", "") or ""
    duration = entry.get("duration") or 0
    if isinstance(song_names, str):
        song_names = [song_names] if song_names else []

    # 1. Video sudah dipakai lagu lain (atau video lama yang ditolak saat scrape ulang)
    if blocked_ids and entry.get("id") in blocked_ids:
        return False, "Video sudah dipakai lagu lain / ditolak sebelumnya", 0.0

    # 2. Validasi Durasi
    # Lagu daerah resmi biasanya berdurasi 1 - 7 menit (65 - 420 detik)
    if duration > 0 and (duration < 65 or duration > 420):
        return False, f"Durasi tidak lazim ({duration} detik)", 0.0

    # 3. Validasi Blacklist (Anak-anak, Kartun, Remix, Berita, Medley, dll)
    #    Kata yang merupakan bagian nama lagu (mis. 'Anak Medan', 'Tari Bali') tidak dihitung.
    kata_nama = {t for n in song_names for t in _token(n)}
    combined_text = f"{title} {channel} {desc[:200]}"
    for blacklist_match in BLACKLIST_REGEX.finditer(combined_text):
        if blacklist_match.group(0).lower() not in kata_nama:
            return False, f"Kata kunci terlarang: '{blacklist_match.group(0)}'", 0.0

    # 4. Judul harus memuat nama lagu (nama utama atau nama lain)
    skor = max((kemiripan_judul(n, title) for n in song_names), default=1.0)
    if skor < MIN_KEMIRIPAN_JUDUL:
        return False, f"Judul tidak cocok dengan nama lagu ({skor:.2f})", skor

    return True, "Valid", skor


# Alias untuk kompatibilitas ke belakang
evaluate_candidate = validate_candidate


def skor_kandidat(entry: dict, kemiripan: float, province: str) -> float:
    """Skor peringkat kandidat yang sudah lolos validasi: judul > label resmi > provinsi di judul."""
    title = (entry.get("title") or "").lower()
    channel = (entry.get("channel") or entry.get("uploader") or "").lower()
    skor = kemiripan
    if channel.endswith("- topic") or any(lbl in channel for lbl in OFFICIAL_LABELS):
        skor += 0.5
    if province and province.lower() in title:
        skor += 0.1
    return skor


def search_candidates(ydl: yt_dlp.YoutubeDL, query: str, max_results: int = 5) -> list[dict]:
    """Mencari daftar kandidat video tanpa mengunduh kontennya."""
    search_url = f"ytsearch{max_results}:{query}"
    try:
        res = ydl.extract_info(search_url, download=False)
        if res and "entries" in res:
            return [e for e in res["entries"] if e]
        return []
    except Exception as err:
        print(f"    [!] Error saat pencarian: {err}", file=sys.stderr)
        return []


def baca_ids_cek_manual(path: str) -> set[str]:
    """Ambil id lagu dengan keputusan 'scrape ulang' dari outputs/eda/daftar_cek_manual.csv."""
    with open(path, encoding="utf-8-sig") as f:
        return {
            r["id"].strip() for r in csv.DictReader(f)
            if (r.get("keputusan") or "").strip().lower().replace("_", " ") == "scrape ulang"
        }


MANIFEST_FIELDS = [
    "id", "nama_lagu", "wilayah", "provinsi",
    "youtube_id", "video_title", "channel",
    "duration_sec", "audio_path", "status", "reasons"
]


def tulis_manifest(path: str, rows: dict[str, dict]) -> None:
    """Tulis ulang manifest (satu baris per id lagu, urut id) secara atomik."""
    tmp = path + ".tmp"
    with open(tmp, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for _, row in sorted(rows.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9):
            writer.writerow(row)
    os.replace(tmp, path)


def process_songs(
    csv_file: str,
    output_dir: str,
    limit: int = 5,
    start_id: int = 1,
    audio_format: str = "mp3",
    snippet_start: int = 0,
    snippet_duration: int = 0,
    dry_run: bool = False,
    verbose: bool = False,
    ids: set[str] | None = None,
    force: bool = False,
) -> None:
    """Memproses kurasi dan pengunduhan lagu dari CSV."""
    csv_file = resolve_project_path(csv_file)
    output_dir = resolve_project_path(output_dir)

    ffmpeg_path = resolve_ffmpeg()
    if not dry_run and not ffmpeg_path:
        print("[!] PERINGATAN: ffmpeg tidak ditemukan. Konversi audio mungkin gagal.", file=sys.stderr)
    else:
        if verbose and ffmpeg_path:
            print(f"[*] ffmpeg terdeteksi di: {ffmpeg_path}")

    # Baca dataset CSV
    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter target berdasarkan ids (jika ada), atau limit dan start_id
    if ids:
        target_rows = [r for r in all_rows if r.get("id", "").strip() in ids]
    else:
        target_rows = [r for r in all_rows if int(r.get("id", 0)) >= start_id]
        if limit > 0:
            target_rows = target_rows[:limit]

    print(f"[*] Memproses {len(target_rows)} lagu (Start ID: {start_id}, Limit: {limit}, Force: {force})...")
    print(f"[*] Mode: {'DRY RUN (Pencarian & Validasi Saja)' if dry_run else 'LIVE DOWNLOAD'}")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "audio_manifest.csv")

    # Muat manifest lama (satu baris per id; baris terakhir yang berlaku)
    manifest_rows: dict[str, dict] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                manifest_rows[r["id"]] = r

    # Dry run tidak boleh mengotori manifest utama -> tulis ke file terpisah
    out_manifest_path = os.path.join(output_dir, "audio_manifest_dryrun.csv") if dry_run else manifest_path
    out_rows = {} if dry_run else manifest_rows

    # Backup manifest sebelum diubah
    if not dry_run and os.path.exists(manifest_path):
        backup_dir = os.path.join(output_dir, "backup_manifest")
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(
            backup_dir, f"audio_manifest.backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        shutil.copy2(manifest_path, backup)
        print(f"[*] Backup manifest: {backup}")

    # Peta youtube_id -> id lagu yang sudah memakainya (anti-duplikat)
    pemakai_video = {
        r["youtube_id"]: r["id"] for r in manifest_rows.values()
        if r.get("youtube_id") and r.get("status") == "DOWNLOADED"
    }

    # Inisialisasi yt-dlp options untuk pencarian (flat)
    search_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(search_opts) as ydl_search:
        for idx, song in enumerate(target_rows, 1):
            song_id = song["id"]
            nama_lagu = song["nama_lagu"]
            provinsi = song["provinsi"]
            wilayah = song["wilayah"]
            nama_semua = daftar_nama(song)

            print(f"\n[{idx}/{len(target_rows)}] #{song_id} - {nama_lagu} ({provinsi}, {wilayah})")

            # Tentukan subfolder per provinsi untuk keteraturan data
            provinsi_dir = os.path.join(output_dir, sanitize_filename(provinsi))
            os.makedirs(provinsi_dir, exist_ok=True)
            clean_song_slug = sanitize_filename(f"{song_id}_{nama_lagu}")
            final_audio_path = os.path.join(provinsi_dir, f"{clean_song_slug}.{audio_format}")

            # Cek jika berkas sudah ada sebelumnya (Resume support), kecuali --force
            if not force and not dry_run and os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 10000:
                print(f"  [>] Lagu sudah terunduh sebelumnya: {final_audio_path} (Dilewati)")
                continue

            # Video yang diblok: dipakai lagu lain + video lama lagu ini bila scrape ulang
            video_lama = (manifest_rows.get(song_id) or {}).get("youtube_id", "")
            blocked = {v for v, pemilik in pemakai_video.items() if pemilik != song_id}
            if force and video_lama:
                blocked.add(video_lama)

            # Susun variasi query pencarian yang memprioritaskan studio release
            queries = [
                f'"{nama_lagu}" {provinsi} "Topic"',
                f'"{nama_lagu}" {provinsi} "GNP Music"',
                f'"{nama_lagu}" {provinsi} lagu daerah official audio',
                f'"{nama_lagu}" {provinsi} lagu daerah'
            ]

            best_candidate, best_score, best_sim = None, -1.0, 0.0
            reject_reasons = []
            sudah_dicek = set()

            for q in queries:
                if verbose:
                    print(f"  -> Mencari: {q}")
                candidates = search_candidates(ydl_search, q, max_results=5)

                for cand in candidates:
                    if cand.get("id") in sudah_dicek:
                        continue
                    sudah_dicek.add(cand.get("id"))
                    is_valid, reason, sim = validate_candidate(cand, nama_semua, provinsi, blocked)
                    if verbose:
                        status_label = f"VALID {sim:.2f}" if is_valid else f"INVALID: {reason}"
                        print(f"     * [{status_label}] {cand.get('title')} ({cand.get('channel')})")
                    if not is_valid:
                        reject_reasons.append(reason)
                        continue
                    skor = skor_kandidat(cand, sim, provinsi)
                    if skor > best_score:
                        best_candidate, best_score, best_sim = cand, skor, sim

                # Berhenti lebih awal jika sudah dapat judul persis dari channel resmi
                if best_score >= 1.5:
                    break

            if not best_candidate:
                print(f"  [-] Tidak ada kandidat yang memenuhi kualifikasi (Lagu dilewati).")
                lama = manifest_rows.get(song_id)
                if not dry_run and lama and lama.get("status") == "DOWNLOADED" and os.path.exists(lama.get("audio_path", "")):
                    print(f"      Data lama dipertahankan: {lama.get('audio_path')}")
                    continue
                out_rows[song_id] = {
                    "id": song_id,
                    "nama_lagu": nama_lagu,
                    "wilayah": wilayah,
                    "provinsi": provinsi,
                    "youtube_id": "",
                    "video_title": "",
                    "channel": "",
                    "duration_sec": 0,
                    "audio_path": "",
                    "status": "SKIPPED_NO_QUALIFIED_CANDIDATE",
                    "reasons": "; ".join(sorted(set(reject_reasons))) if reject_reasons else "Tidak ditemukan kandidat yang valid"
                }
                tulis_manifest(out_manifest_path, out_rows)
                continue

            vid_id = best_candidate.get("id")
            vid_title = best_candidate.get("title")
            vid_channel = best_candidate.get("channel") or best_candidate.get("uploader")
            vid_duration = best_candidate.get("duration") or 0
            vid_url = f"https://www.youtube.com/watch?v={vid_id}"
            alasan = f"Lolos validasi; kemiripan judul {best_sim:.2f}; skor {best_score:.2f}"

            print(f"  [+] Terpilih: {vid_title}")
            print(f"      Channel: {vid_channel} | Durasi: {vid_duration}s | {alasan}")

            if dry_run:
                print(f"      [DRY-RUN] Melewati proses download audio.")
                out_rows[song_id] = {
                    "id": song_id,
                    "nama_lagu": nama_lagu,
                    "wilayah": wilayah,
                    "provinsi": provinsi,
                    "youtube_id": vid_id,
                    "video_title": vid_title,
                    "channel": vid_channel,
                    "duration_sec": vid_duration,
                    "audio_path": manifest_audio_path(final_audio_path),
                    "status": "DRY_RUN_MATCHED",
                    "reasons": alasan
                }
                tulis_manifest(out_manifest_path, out_rows)
                continue

            # Konfigurasi pengunduhan audio menggunakan yt-dlp
            postprocessors = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }]

            # Opsi pemotongan cuplikan (snippet) jika ditentukan
            download_ranges = None
            if snippet_duration > 0:
                start_s = snippet_start
                end_s = snippet_start + snippet_duration
                download_ranges = yt_dlp.utils.download_range_func(None, [(start_s, end_s)])

            # Unduh ke nama sementara dulu agar file lama tidak hilang bila unduhan gagal
            tmp_slug = f"{clean_song_slug}.baru"
            tmp_audio_path = os.path.join(provinsi_dir, f"{tmp_slug}.{audio_format}")
            download_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(provinsi_dir, f"{tmp_slug}.%(ext)s"),
                "postprocessors": postprocessors,
                "ffmpeg_location": ffmpeg_path,
                "quiet": not verbose,
                "no_warnings": True,
                "overwrites": True,
            }
            if download_ranges:
                download_opts["download_ranges"] = download_ranges

            try:
                print(f"  [*] Mengunduh & mengekstrak audio ({audio_format.upper()})...")
                with yt_dlp.YoutubeDL(download_opts) as ydl_down:
                    ydl_down.download([vid_url])
                os.replace(tmp_audio_path, final_audio_path)

                status = "DOWNLOADED"
                print(f"  [OK] Tersimpan: {final_audio_path}")
            except Exception as dl_err:
                print(f"  [!] Gagal mengunduh: {dl_err}", file=sys.stderr)
                status = f"FAILED: {dl_err}"
                if force and song_id in manifest_rows and os.path.exists(final_audio_path):
                    print(f"      Data lama dipertahankan: {final_audio_path}")
                    continue

            out_rows[song_id] = {
                "id": song_id,
                "nama_lagu": nama_lagu,
                "wilayah": wilayah,
                "provinsi": provinsi,
                "youtube_id": vid_id,
                "video_title": vid_title,
                "channel": vid_channel,
                "duration_sec": vid_duration,
                "audio_path": manifest_audio_path(final_audio_path) if status == "DOWNLOADED" else "",
                "status": status,
                "reasons": alasan if status == "DOWNLOADED" else status
            }
            if status == "DOWNLOADED":
                pemakai_video = {v: p for v, p in pemakai_video.items() if p != song_id}
                pemakai_video[vid_id] = song_id
            tulis_manifest(out_manifest_path, out_rows)

            # Jeda etis antar permintaan
            time.sleep(1.0)

    print("\n" + "=" * 65)
    print(f" [SELESAI] Hasil tercatat di: {out_manifest_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Scraper Audio Musik Tradisional Profesional dari YouTube untuk Deep Learning"
    )
    parser.add_argument(
        "--csv",
        default="data/daftar_lagu_daerah.csv",
        help="Path ke file dataset CSV (default: data/daftar_lagu_daerah.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Direktori target penyimpanan data provinsi (default: data)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Jumlah lagu yang diproses (default: 5 untuk trial; 0 untuk semua)"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="ID awal lagu yang diproses (default: 1)"
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Daftar id lagu dipisah koma, mis. 12,45,78 (mengabaikan --limit/--start-id)"
    )
    parser.add_argument(
        "--dari-cek-manual",
        default="",
        help="Path daftar_cek_manual.csv; proses semua id dengan keputusan 'scrape ulang'"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Unduh ulang walau file sudah ada; video lama lagu tsb tidak akan dipilih lagi"
    )
    parser.add_argument(
        "--min-kemiripan",
        type=float,
        default=0.6,
        help="Ambang kecocokan judul video vs nama lagu, 0-1 (default: 0.6)"
    )
    parser.add_argument(
        "--format",
        choices=["m4a", "wav", "mp3", "flac"],
        default="m4a",
        help="Format konversi audio: m4a, wav, mp3, flac (default: m4a, efisien disk ~2.5GB & kompatibel Librosa/PyTorch)"
    )
    parser.add_argument(
        "--snippet-start",
        type=int,
        default=0,
        help="Detik mulai cuplikan audio (0 = lagu utuh)"
    )
    parser.add_argument(
        "--snippet-duration",
        type=int,
        default=0,
        help="Durasi cuplikan audio dalam detik (0 = lagu utuh)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya jalankan pencarian dan validasi kelayakan tanpa mengunduh berkas"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Tampilkan log terperinci dari yt-dlp"
    )

    args = parser.parse_args()

    # Reconfigure UTF-8 encoding untuk terminal Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    global MIN_KEMIRIPAN_JUDUL
    MIN_KEMIRIPAN_JUDUL = args.min_kemiripan

    ids = {i.strip() for i in args.ids.split(",") if i.strip()}
    if args.dari_cek_manual:
        ids |= baca_ids_cek_manual(resolve_project_path(args.dari_cek_manual))
        if not ids:
            print("[!] Tidak ada baris dengan keputusan 'scrape ulang'.")
            return

    process_songs(
        csv_file=args.csv,
        output_dir=args.output_dir,
        limit=args.limit,
        start_id=args.start_id,
        audio_format=args.format,
        snippet_start=args.snippet_start,
        snippet_duration=args.snippet_duration,
        dry_run=args.dry_run,
        verbose=args.verbose,
        ids=ids or None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
