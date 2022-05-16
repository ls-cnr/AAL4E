import source_state_check
import sys

try:
    if source_state_check.idle_recognition() == 1:
        print("Idle Recognised")
    else:
        print("Error")
except KeyboardInterrupt:
    sys.exit(0)
 