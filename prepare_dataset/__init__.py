from .get_youtube import get_audios
from .do_spleeter import get_vocals
from .labeled_midi import get_midis
from .get_cqt import get_cqt_feature

def prepare_dataset():
    get_audios()
    get_vocals()
    get_midis()
    get_cqt_feature()
