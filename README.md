# PythonAudio
Audio DSP using Python
The aim of this repository is to create a comprehensive, curated list of python software/tools related and used for scientific research in audio/music applications

## Contents
## Contents

* [Audio Related Packages](#audio-related-packages)
    - [Read/Write](#read-write)
    - [Transformations - General DSP](#transformations---general-dsp)
    - [Feature extraction](#feature-extraction)
    - [Data augmentation](#data-augmentation)
    - [Speech Processing](#speech-processing)
    - [Environmental Sounds](#environmenta)
    - [Perceptial Models - Auditory Models](#perceptial-models---auditory-models)
    - [Source Separation](#source-separation)
    - [Music Information Retrieval](#music-information-retrieval)
    - [Deep Learning](#deep-learning)
    - [Symbolic Music - MIDI - Musicology](#symbolic-music---midi---musicology)
    - [Realtime applications](#realtime-applications)
    - [Web - Audio](#web-audio)
    - [Audio related APIs and Datasets](#audio-related-apis-and-datasets)
    - [Wrappers for Audio Plugins](#wrappers-for-audio-plugins)
* [Tutorials](#tutorials)
* [Books](#books)
* [Scientific Paper](#scientific-papers)
* [Other Resources](#other-resources)
* [Related lists](#related-lists)
* [Contributing](#contributing)
* [License](#license)


## Audio Related Packages

- Total number of packages: 66

#### Read-Write

* [audiolazy](https://github.com/danilobellini/audiolazy) (https://github.com/danilobellini/audiolazy) (https://pypi.python.org/pypi/audiolazy/) - Expressive Digital Signal Processing (DSP) package for Python.
* [audioread](https://github.com/beetbox/audioread) (https://github.com/beetbox/audioread) (https://pypi.python.org/pypi/audioread/) - Cross-library (GStreamer + Core Audio + MAD + FFmpeg) audio decoding.
* [mutagen](https://mutagen.readthedocs.io/) (https://github.com/quodlibet/mutagen) (https://pypi.python.org/pypi/mutagen) - Reads and writes all kind of audio metadata for various formats.
* [pyAV](http://docs.mikeboers.com/pyav/) (https://github.com/mikeboers/PyAV) - PyAV is a Pythonic binding for FFmpeg or Libav.
* [(Py)Soundfile](http://pysoundfile.readthedocs.io/) (https://github.com/bastibe/PySoundFile) (https://pypi.python.org/pypi/SoundFile) - Library based on libsndfile, CFFI, and NumPy.

#### Transformations - General DSP

* [acoustics](http://python-acoustics.github.io/python-acoustics/) (https://github.com/python-acoustics/python-acoustics/) (https://pypi.python.org/pypi/acoustics) - useful tools for acousticians.
* [AudioTK](https://github.com/mbrucher/AudioTK) (https://github.com/mbrucher/AudioTK) - DSP filter toolbox (lots of filters).
* [AudioTSM](https://audiotsm.readthedocs.io/) (https://github.com/Muges/audiotsm) (https://pypi.python.org/pypi/audiotsm/) - real-time audio time-scale modification procedures.
* [Gammatone](https://github.com/detly/gammatone) (https://github.com/detly/gammatone) - Gammatone filterbank implementation.
* [pyFFTW](http://pyfftw.github.io/pyFFTW/) (https://github.com/pyFFTW/pyFFTW) (https://pypi.python.org/pypi/pyFFTW/) - Wrapper for FFTW(3).
* [NSGT](https://grrrr.org/research/software/nsgt/) (https://github.com/grrrr/nsgt) (https://pypi.python.org/pypi/nsgt) - Non-stationary gabor transform, constant-q.
* [matchering](https://github.com/sergree/matchering) (https://github.com/sergree/matchering) (https://pypi.org/project/matchering/) - Automated reference audio mastering.
* [MDCT](https://github.com/nils-werner/mdct) (https://github.com/nils-werner/mdct) (https://pypi.python.org/pypi/mdct) - MDCT transform.
* [pydub](http://pydub.com) (https://github.com/jiaaro/pydub) (https://pypi.python.org/pypi/mdct) - Manipulate audio with a simple and easy high level interface.
* [pytftb](http://tftb.nongnu.org) (https://github.com/scikit-signal/pytftb) - Implementation of the MATLAB Time-Frequency Toolbox.
* [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (https://github.com/LCAV/pyroomacoustics) (https://pypi.python.org/pypi/pyroomacoustics) - Room Acoustics Simulation (RIR generator)
* [PyRubberband](https://github.com/bmcfee/pyrubberband) (https://github.com/bmcfee/pyrubberband) (https://pypi.python.org/pypi/pyrubberband/) - Wrapper for [rubberband](http://breakfastquay.com/rubberband/) to do pitch-shifting and time-stretching.
* [PyWavelets](http://pywavelets.readthedocs.io) (https://github.com/PyWavelets/pywt) (https://pypi.python.org/pypi/PyWavelets) - Discrete Wavelet Transform in Python.
* [Resampy](http://resampy.readthedocs.io) (https://github.com/bmcfee/resampy) (https://pypi.python.org/pypi/resampy) - Sample rate conversion.
* [SFS-Python](http://www.sfstoolbox.org) (https://github.com/sfstoolbox/sfs-python) (https://pypi.python.org/pypi/sfs/) - Sound Field Synthesis Toolbox.
* [sound_field_analysis](https://appliedacousticschalmers.github.io/sound_field_analysis-py/) (https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py) (https://pypi.org/project/sound-field-analysis/) - Analyze, visualize and process sound field data recorded by spherical microphone arrays.
* [STFT](http://stft.readthedocs.io) (https://github.com/nils-werner/stft) (https://pypi.python.org/pypi/stft) - Standalone package for Short-Time Fourier Transform.

#### Feature extraction

* [aubio](http://aubio.org/) (https://github.com/aubio/aubio) (https://pypi.python.org/pypi/aubio) - Feature extractor, written in C, Python interface.
* [audioFlux](https://github.com/libAudioFlux/audioFlux) (https://github.com/libAudioFlux/audioFlux) (https://pypi.python.org/pypi/audioflux) - A library for audio and music analysis, feature extraction.
* [audiolazy](https://github.com/danilobellini/audiolazy) (https://github.com/danilobellini/audiolazy) (https://pypi.python.org/pypi/audiolazy/) - Realtime Audio Processing lib, general purpose.
* [essentia](http://essentia.upf.edu) (https://github.com/MTG/essentia) - Music related low level and high level feature extractor, C++ based, includes Python bindings.
* [python_speech_features](https://github.com/jameslyons/python_speech_features) (https://github.com/jameslyons/python_speech_features) (https://pypi.python.org/pypi/python_speech_features) - Common speech features for ASR.
* [pyYAAFE](https://github.com/Yaafe/Yaafe) (https://github.com/Yaafe/Yaafe) - Python bindings for YAAFE feature extractor.
* [speechpy](https://github.com/astorfi/speechpy) (https://github.com/astorfi/speechpy) (https://pypi.python.org/pypi/speechpy) - Library for Speech Processing and Recognition, mostly feature extraction for now.
* [spafe](https://github.com/SuperKogito/spafe) (https://github.com/SuperKogito/spafe) (https://pypi.org/project/spafe/) - Python library for features extraction from audio files.


#### Speech Processing

* [aeneas](https://www.readbeyond.it/aeneas/) (https://github.com/readbeyond/aeneas/) (https://pypi.python.org/pypi/aeneas/) - Forced aligner, based on MFCC+DTW, 35+ languages.
* [deepspeech](https://github.com/mozilla/DeepSpeech) (https://github.com/mozilla/DeepSpeech) (https://pypi.org/project/deepspeech/) - Pretrained automatic speech recognition.
* [gentle](https://github.com/lowerquality/gentle) (https://github.com/lowerquality/gentle) - Forced-aligner built on Kaldi.
* [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (https://github.com/YannickJadoul/Parselmouth) (https://pypi.org/project/praat-parselmouth/) - Python interface to the [Praat](http://www.praat.org) phonetics and speech analysis, synthesis, and manipulation software.
* [persephone](https://persephone.readthedocs.io/en/latest/) (https://github.com/persephone-tools/persephone) (https://pypi.org/project/persephone/) - Automatic phoneme transcription tool.
* [pyannote.audio](https://github.com/pyannote/pyannote-audio) (https://github.com/pyannote/pyannote-audio) (https://pypi.org/project/pyannote-audio/) - Neural building blocks for speaker diarization.
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)² (https://github.com/tyiannak/pyAudioAnalysis) (https://pypi.python.org/pypi/pyAudioAnalysis/) - Feature Extraction, Classification, Diarization.
* [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) (https://github.com/wiseman/py-webrtcvad) (https://pypi.python.org/pypi/webrtcvad/) -  Interface to the WebRTC Voice Activity Detector.
* [pypesq](https://github.com/vBaiCai/python-pesq) (https://github.com/vBaiCai/python-pesq) - Wrapper for the PESQ score calculation.
* [pystoi](https://github.com/mpariente/pystoi) (https://github.com/mpariente/pystoi) (https://pypi.org/project/pystoi) - Short Term Objective Intelligibility measure (STOI).
* [PyWorldVocoder](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) (https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - Wrapper for Morise's World Vocoder.
* [Montreal Forced Aligner](https://montrealcorpustools.github.io/Montreal-Forced-Aligner/) (https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) - Forced aligner, based on Kaldi (HMM), English (others can be trained).
* [SIDEKIT](http://lium.univ-lemans.fr/sidekit/) (https://pypi.python.org/pypi/SIDEKIT/) - Speaker and Language recognition.
* [SpeechRecognition](https://github.com/Uberi/speech_recognition) (https://github.com/Uberi/speech_recognition) (https://pypi.python.org/pypi/SpeechRecognition/) -  Wrapper for several ASR engines and APIs, online and offline.

#### Environmental Sounds

* [sed_eval](http://tut-arg.github.io/sed_eval) (https://github.com/TUT-ARG/sed_eval) (https://pypi.org/project/sed_eval/) - Evaluation toolbox for Sound Event Detection

#### Perceptial Models - Auditory Models

* [cochlea](https://github.com/mrkrd/cochlea) (https://github.com/mrkrd/cochlea) (https://pypi.python.org/pypi/cochlea/) - Inner ear models.
* [Brian2](http://briansimulator.org/) (https://github.com/brian-team/brian2) (https://pypi.python.org/pypi/Brian2) - Spiking neural networks simulator, includes cochlea model.
* [Loudness](https://github.com/deeuu/loudness) (https://github.com/deeuu/loudness) - Perceived loudness, includes Zwicker, Moore/Glasberg model.
* [pyloudnorm](https://www.christiansteinmetz.com/projects-blog/pyloudnorm) (https://github.com/csteinmetz1/pyloudnorm) - Audio loudness meter and normalization, implements ITU-R BS.1770-4.
* [Sound Field Synthesis Toolbox](http://www.sfstoolbox.org) (https://github.com/sfstoolbox/sfs-python) (https://pypi.python.org/pypi/sfs/) - Sound Field Synthesis Toolbox.

#### Source Separation

* [commonfate](https://github.com/aliutkus/commonfate) (https://github.com/aliutkus/commonfate) (https://pypi.python.org/pypi/commonfate) - Common Fate Model and Transform.
* [NTFLib](https://github.com/stitchfix/NTFLib) (https://github.com/stitchfix/NTFLib) - Sparse Beta-Divergence Tensor Factorization.
* [NUSSL](https://interactiveaudiolab.github.io/project/nussl.html) (https://github.com/interactiveaudiolab/nussl) (https://pypi.python.org/pypi/nussl) - Holistic source separation framework including DSP methods and deep learning methods.
* [NIMFA](http://nimfa.biolab.si) (https://github.com/marinkaz/nimfa) (https://pypi.python.org/pypi/nimfa) - Several flavors of non-negative-matrix factorization.

#### Wrappers for Audio Plugins

* [VamPy Host](https://code.soundsoftware.ac.uk/projects/vampy-host) (https://pypi.python.org/pypi/vamp) - Interface compiled vamp plugins.

## Tutorials

* [Whirlwind Tour Of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) (https://github.com/jakevdp/WhirlwindTourOfPython
) - fast-paced introduction to Python essentials, aimed at researchers and developers.
* [Introduction to Numpy and Scipy](http://www.scipy-lectures.org/index.html) (https://github.com/scipy-lectures/scipy-lecture-notes) - Highly recommended tutorial, covers large parts of the scientific Python ecosystem.
* [Numpy for MATLAB® Users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html) - Short overview of equivalent python functions for switchers.
* [MIR Notebooks](http://musicinformationretrieval.com/) (https://github.com/stevetjoa/stanford-mir) - collection of instructional iPython Notebooks for music information retrieval (MIR).
* [Selected Topics in Audio Signal Processing]( https://github.com/spatialaudio/selected-topics-in-audio-signal-processing-exercises) - Exercises as iPython notebooks.
* [Live-coding a music synthesizer](https://www.youtube.com/watch?v=SSyQ0kRHzis) Live-coding video showing how to use the SoundDevice library to reproduce realistic sounds. [Code](https://github.com/cool-RR/python_synthesizer).

## Books

* [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) - Jake Vanderplas, Excellent Book and accompanying tutorial notebooks.
* [Fundamentals of Music Processing](https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP) - Meinard Müller, comes with Python exercises.

## Scientific Papers

* [Python for audio signal processing](http://eprints.maynoothuniversity.ie/4115/1/40.pdf) - John C. Glover, Victor Lazzarini and Joseph Timoney, Linux Audio Conference 2011.
* [librosa: Audio and Music Signal Analysis in Python](http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf), [Video](https://www.youtube.com/watch?v=MhOdbtPhbLU) - Brian McFee, Colin Raffel, Dawen Liang, Daniel P.W. Ellis, Matt McVicar, Eric Battenberg, Oriol Nieto, Scipy 2015.
* [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255), [Video](https://www.youtube.com/watch?v=37R_R82lfwA) - Hervé Bredin, Ruiqing Yin, Juan Manuel Coria, Gregory Gelly, Pavel Korshunov, Marvin Lavechin, Diego Fustes, Hadrien Titeux, Wassim Bouaziz, Marie-Philippe Gill, ICASSP 2020.


