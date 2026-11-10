# YouTube のライブ配信を指定秒数だけ mp4 に保存する（再エンコードなし、音なし）
#   python record.py <YouTube の URL> <出力.mp4> [秒数=60] [ffmpeg のパス]
# 配信の「今」から読むと HLS の更新待ちで止まることがあったので、
# プレイリストの少し前（-live_start_index -14 ≒ 70 秒前）から読む。
import subprocess, sys
url, out = sys.argv[1], sys.argv[2]
secs = sys.argv[3] if len(sys.argv) > 3 else "60"
ffmpeg = sys.argv[4] if len(sys.argv) > 4 else "ffmpeg"
m3u8 = subprocess.check_output(["yt-dlp", "--no-update", "-f", "95", "-g", url], text=True).split()[0]
subprocess.check_call([ffmpeg, "-v", "warning", "-y", "-live_start_index", "-14", "-i", m3u8,
                       "-t", secs, "-c", "copy", "-an", out])
print("saved", out)
