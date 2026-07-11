# vhs tape for docs/resume-screenshot.png — memo's cross-agent resume picker.
#
# Regenerate (installs: brew install vhs imagemagick):
#   vhs docs/resume.tape                                   # writes /tmp/memo-resume.gif
#   magick /tmp/memo-resume.gif -coalesce coal-%04d.png    # split to full frames
#   magick "$(ls coal-*.png | tail -1)" -crop 1240x492+0+0 +repage docs/resume-screenshot.png
#   rm coal-*.png /tmp/memo-resume.gif
#
# The picker (`memo resume`) scans the operator's real, private sessions, so the
# screenshot is driven through docs/resume-demo.py — the *actual* interactive TUI
# rendered with public, illustrative sample sessions instead of real ones.
Output "/tmp/memo-resume.gif"

Set Shell "bash"
Set FontSize 18
Set Width 1240
Set Height 720
Set Padding 26
Set Theme "Dracula"

Hide
Type 'export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"'
Enter
Type 'cd "$HOME/repos/memo"'
Enter
Type 'export MEMO_DATA_DIR=/tmp/memo-resume-demo MEMO_STATE_DIR=/tmp/memo-resume-demo-state'
Enter
Type 'clear'
Enter
Show

Type 'uv run --no-sync python docs/resume-demo.py'
Enter
Sleep 6s
Down@350ms 1
Sleep 2s
