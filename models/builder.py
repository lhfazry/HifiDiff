from models.diffwave import DiffWave
from models.hifidiff import HifiDiff
from models.hifidiffv2 import HifiDiffV2
from models.hifidiffv3 import HifiDiffV3
from models.hifidiffv4 import HifiDiffV4
from models.hifidiffv5 import HifiDiffV5
from models.hifidiffv7 import HifiDiffV7
from models.hifidiffv7r1 import HifiDiffV7R1
from models.hifidiffv8 import HifiDiffV8
from models.wavegrad import WaveGrad

def build_model(params):
    if params.model == 1:
        model = DiffWave(params)
    elif params.model == 2:
        model = HifiDiff(params)
    elif params.model == 3:
        model = HifiDiffV2(params)
    elif params.model == 4:
        model = HifiDiffV3(params)
    elif params.model == 5:
        model = HifiDiffV4(params)
    elif params.model == 6:
        model = HifiDiffV5(params)
    elif params.model == 7:
        model = HifiDiffV7(params)
    elif params.model == 8:
        model = HifiDiffV7R1(params)
    elif params.model == 9:
        model = HifiDiffV8(params)
    elif params.model == 99:
        model = WaveGrad(params)

    #assert model is None

    return model