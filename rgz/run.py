import os
import sys

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
    sys.exec_prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins'
)

from main import main

if __name__ == '__main__':
    main()