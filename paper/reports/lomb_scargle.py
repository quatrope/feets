
import sys
import time as tmod
import warnings

import numpy as np

warnings.simplefilter("ignore")

sys.path.insert(0, "../FATS/")
import FATS


#We open the ligth curve in two different bands
lc_B = FATS.ReadLC_MACHO('lc/lc_1.3444.614.B.txt')
lc_R = FATS.ReadLC_MACHO('lc/lc_1.3444.614.R.txt')

#We import the data
[mag, time, error] = lc_B.ReadLC()
[mag2, time2, error2] = lc_R.ReadLC()

#We preprocess the data
preproccesed_data = FATS.Preprocess_LC(mag, time, error)
[mag, time, error] = preproccesed_data.Preprocess()

preproccesed_data = FATS.Preprocess_LC(mag2, time2, error2)
[mag2, time2, error2] = preproccesed_data.Preprocess()

#We synchronize the data
if len(mag) != len(mag2):
    [aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2] = \
    FATS.Align_LC(time, time2, mag, mag2, error, error2)

lc = np.array([mag, time, error, mag2, aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2])

EXCLUDE = [
    'Freq1_harmonics_amplitude_0','Freq1_harmonics_amplitude_1',
    'Freq1_harmonics_amplitude_2','Freq1_harmonics_amplitude_3',
    'Freq2_harmonics_amplitude_0','Freq2_harmonics_amplitude_1',
    'Freq2_harmonics_amplitude_2','Freq2_harmonics_amplitude_3',
    'Freq3_harmonics_amplitude_0','Freq3_harmonics_amplitude_1',
    'Freq3_harmonics_amplitude_2','Freq3_harmonics_amplitude_3',
    'Freq1_harmonics_amplitude_0','Freq1_harmonics_rel_phase_0',
    'Freq1_harmonics_rel_phase_1','Freq1_harmonics_rel_phase_2',
    'Freq1_harmonics_rel_phase_3','Freq2_harmonics_rel_phase_0',
    'Freq2_harmonics_rel_phase_1','Freq2_harmonics_rel_phase_2',
    'Freq2_harmonics_rel_phase_3','Freq3_harmonics_rel_phase_0',
    'Freq3_harmonics_rel_phase_1','Freq3_harmonics_rel_phase_2',
    'Freq3_harmonics_rel_phase_3', "Period_fit", "Psi_eta", "Psi_CS"]


iterations = 1000

times_pls = []
fs = FATS.FeatureSpace(
    Data='all', excludeList=EXCLUDE)

for _ in range(iterations):
    start = tmod.time()
    fs.calculateFeature(lc)
    times_pls.append(tmod.time() - start)


times = []
fs = FATS.FeatureSpace(
    Data='all', excludeList=EXCLUDE + ["PeriodLS"])

for _ in range(iterations):
    start = tmod.time()
    fs.calculateFeature(lc)
    times.append(tmod.time() - start)

msg = """
Total iterations: {iterations}
With PeriodLS:
    - Total: {total_pls}
    - Minimun: {min_pls}
    - Maximun: {max_pls}
    - Mean: {mean_pls}
    - Std: {std_pls}
Without PeriodLS:
    - Total: {total}
    - Minimun: {min}
    - Maximun: {max}
    - Mean: {mean}
    - Std: {std}
""".format(
    iterations=iterations,

    total_pls=np.sum(times_pls), min_pls=np.min(times_pls),
    max_pls=np.max(times_pls), mean_pls=np.mean(times_pls),
    std_pls=np.std(times_pls),

    total=np.sum(times), min=np.min(times),
    max=np.max(times), mean=np.mean(times),
    std=np.std(times))

with open("lombscargle_test.txt", "w") as fp:
    fp.write(msg)
