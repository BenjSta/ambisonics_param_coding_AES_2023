import dawdreamer as daw
import numpy as np
from spaudiopy.sph import n3d_to_sn3d, sn3d_to_n3d

class HarpexUpmixerWrapper():
    '''
    A simple VST wrapper for the Harpex upmix plugin using Dawdreamer
    '''

    def __init__(self, fs, plugin_path):
        self.engine = daw.RenderEngine(fs, 512)
        self.fs = fs
        self.harpex_plugin = \
            self.engine.make_plugin_processor("harpex", plugin_path)
        print('Please verify settings and close window. Make sure to set input and output format to Ambix.')
        self.harpex_plugin.open_editor()

    def process(self, foa): 
        inp = self.engine.make_playback_processor("in", n3d_to_sn3d(foa.T))

        graph = [
            (inp, []),
            (self.harpex_plugin, [inp.get_name()]),
        ]

        self.engine.load_graph(graph)
        self.engine.render(foa.shape[0] / self.fs + 0.5)
        hoa = sn3d_to_n3d(self.engine.get_audio()).T
        
        foa_w = foa[:, 0]
        hoa_w = hoa[:, 0]

        # delay compensation using cross-correlation
        cc = np.fft.irfft(np.conj(np.fft.rfft(foa_w, 2 * hoa_w.shape[0])) * \
            np.fft.rfft(hoa_w, 2 * hoa_w.shape[0]))

        delay = np.argmax(np.abs(cc))

        hoa = hoa[delay:delay+foa.shape[0], :]

        return hoa