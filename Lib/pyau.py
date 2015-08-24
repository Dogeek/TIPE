# create a sound file in AU format playing a sine wave
# of a given frequency, duration and volume
# vegaseat code modified to work with Python27 and Python32
from struct import pack
from math import sin, pi
def au_file(name='test.au', freq=440, dur=1000, vol=0.5):
    """
    creates an AU format sine wave audio file
    of frequency freq (Hz)
    of duration dur (milliseconds)
    and volume vol (max is 1.0)
    """
    fout = open(name, 'wb')
    # header needs size, encoding=2, sampling_rate=8000, channel=1
    fout.write(pack('>4s5L', '.snd'.encode("utf8"), 24, 8*dur, 2, 8000, 1))
    factor = 2 * pi * freq/8000
    # write data
    for seg in range(8 * dur):
        # sine wave calculations
        sin_seg = sin(seg * factor)
        val = pack('b', int(vol * 127 * sin_seg))
        fout.write(val)
    fout.close()
    print("File %s written" % name)
# test the module ...
if __name__ == '__main__':
    au_file(name='sound440.au', freq=440, dur=2000, vol=0.8)
    # if you have Windows, you can test the audio file
    # otherwise comment this code out
    import os
    os.startfile('sound440.au')
