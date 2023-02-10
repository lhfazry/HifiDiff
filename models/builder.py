from models.diffwave import DiffWave
from models.hifidiff import HifiDiff
from models.hifidiffv2 import HifiDiffV2
from models.hifidiffv3 import HifiDiffV3
from models.hifidiffv4 import HifiDiffV4
from models.hifidiffv5 import HifiDiffV5
from models.hifidiffv7 import HifiDiffV7
from models.hifidiffv7r1 import HifiDiffV7R1
from models.hifidiffv8 import HifiDiffV8
from models.hifidiffv9 import HifiDiffV9
from models.hifidiffv9r1 import HifiDiffV9R1
from models.hifidiffv9r2 import HifiDiffV9R2
from models.hifidiffv9r3 import HifiDiffV9R3
from models.hifidiffv9r4 import HifiDiffV9R4
from models.hifidiffv9r5 import HifiDiffV9R5
from models.hifidiffv9r6 import HifiDiffV9R6
from models.hifidiffv9r7 import HifiDiffV9R7
from models.hifidiffv9r8 import HifiDiffV9R8
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
    elif params.model == 10:
        model = HifiDiffV9(params)
    elif params.model == 11:
        model = HifiDiffV9R1(params)
    elif params.model == 12:
        model = HifiDiffV9R2(params)
    elif params.model == 13:
        model = HifiDiffV9R3(params)
    elif params.model == 14:
        model = HifiDiffV9R4(params)
    elif params.model == 15:
        model = HifiDiffV9R5(params)
    elif params.model == 16:
        model = HifiDiffV9R6(params)
    elif params.model == 17:
        model = HifiDiffV9R7(params)
    elif params.model == 18:
        model = HifiDiffV9R8(params)
    elif params.model == 99:
        model = WaveGrad(params)

    #assert model is None

    return model