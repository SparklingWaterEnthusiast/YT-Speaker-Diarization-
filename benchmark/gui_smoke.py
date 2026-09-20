"""Render the real settings widget without touching production configuration."""
import os
from pathlib import Path
import sys
import json
import time

os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea
from PySide6.QtGui import QFont, QFontDatabase
from ytscribe.config import Config
from ytscribe.ui.settings_dialog import SettingsDialog


def main():
    out=Path('benchmark/results/ui'); out.mkdir(parents=True,exist_ok=True)
    app=QApplication([])
    # Offscreen Qt on Windows may not enumerate native fonts automatically.
    fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    QFontDatabase.addApplicationFont(str(fonts/'segoeui.ttf'))
    app.setFont(QFont('Segoe UI',9))
    dialog=SettingsDialog(Config())
    dialog.show(); app.processEvents()
    if '--probe' in sys.argv:
        dialog._probe_hardware()
        deadline=time.perf_counter()+60
        while dialog._probe is not None and dialog._probe.isRunning():
            app.processEvents(); time.sleep(.01)
            if time.perf_counter()>deadline:
                raise RuntimeError('Hardware probe did not finish in 60 seconds')
        app.processEvents()
        from ytscribe.resources import hardware_inventory, sample
        data=hardware_inventory(detailed=True)
        data['ac_connected']=sample().ac_connected
        (out/'hardware.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(dialog.hardware_label.text())
    group=next(g for g in dialog.findChildren(QGroupBox) if g.title()=='Optimization')
    dialog.findChild(QScrollArea).ensureWidgetVisible(group,0,0)
    app.processEvents()
    if not dialog.grab().save(str(out/'optimization.png')):
        raise RuntimeError('Qt capture failed')
    print('Rendered real Qt settings; torch imported:', 'torch' in sys.modules)
    dialog.reject(); app.processEvents()


if __name__=='__main__': main()
