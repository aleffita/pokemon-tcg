#!/usr/bin/env python3
"""
Final Countdown Trigger Script for Kaggle Pokémon TCG AI Challenge Deadline
Fires 20 minutes before 23:59:59 UTC (at 20:39:59 local UTC-3).
"""

import subprocess
import sys
import time
from datetime import datetime, timezone

def trigger_final_countdown():
    print(f"[{datetime.now().isoformat()}] DEADLINE APPROACHING: 20 MINUTES REMAINING UNTIL SUBMISSION LOCK (23:59:59 UTC)!")
    
    # 1. Open Spotify and play The Final Countdown (Europe)
    applescript_cmd = '''
    try
        tell application "Spotify"
            activate
            open location "spotify:track:3MrRks75PU4y6t7p0QPLaf"
            play
        end tell
    on error
        do shell script "open 'spotify:search:The%20Final%20Countdown'"
    end try
    '''
    try:
        subprocess.run(["osascript", "-e", applescript_cmd], check=False)
        print("Spotify activated with 'The Final Countdown' (Europe).")
    except Exception as e:
        print(f"Could not trigger AppleScript for Spotify: {e}")
        subprocess.run(["open", "https://open.spotify.com/search/The%20Final%20Countdown"], check=False)

if __name__ == "__main__":
    trigger_final_countdown()
