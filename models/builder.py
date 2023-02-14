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
from models.hifidiffv9r9 import HifiDiffV9R9
from models.hifidiffv9r10 import HifiDiffV9R10
from models.hifidiffv9r11 import HifiDiffV9R11
from models.hifidiffv9r12 import HifiDiffV9R12
from models.hifidiffv9r13 import HifiDiffV9R13
from models.hifidiffv9r14 import HifiDiffV9R14
from models.hifidiffv9r15 import HifiDiffV9R15
from models.hifidiffv10 import HifiDiffV10
from models.hifidiffv10r1 import HifiDiffV10R1
from models.hifidiffv10r2 import HifiDiffV10R2
from models.hifidiffv11 import HifiDiffV11
from models.hifidiffv11r1 import HifiDiffV11R1
from models.hifidiffv11r2 import HifiDiffV11R2
from models.hifidiffv11r3 import HifiDiffV11R3
from models.hifidiffv11r4 import HifiDiffV11R4
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
    elif params.model == 19:
        model = HifiDiffV9R9(params)
    elif params.model == 20:
        model = HifiDiffV9R10(params)
    elif params.model == 21:
        model = HifiDiffV9R11(params)
    elif params.model == 22:
        model = HifiDiffV9R12(params)
    elif params.model == 23:
        model = HifiDiffV9R13(params)
    elif params.model == 24:
        model = HifiDiffV9R14(params)
    elif params.model == 25:
        model = HifiDiffV9R15(params)
    elif params.model == 30:
        model = HifiDiffV10(params)
    elif params.model == 31:
        model = HifiDiffV10R1(params)
    elif params.model == 40:
        model = HifiDiffV11(params)
    elif params.model == 41:
        model = HifiDiffV11R1(params)
    elif params.model == 42:
        model = HifiDiffV10R2(params)
    elif params.model == 43:
        model = HifiDiffV11R2(params)
    elif params.model == 44:
        model = HifiDiffV11R3(params)
    elif params.model == 45:
        model = HifiDiffV11R4(params)
    elif params.model == 99:
        model = WaveGrad(params)

    #assert model is None

    return model