import source_state_check
import sys

try:
    if source_state_check.motion_recognition() == 1:
        print("Motion Recognised")
    else:
        print("Error")
except KeyboardInterrupt:
    sys.exit(0)
 