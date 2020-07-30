from contextlib import ContextDecorator
from IPython.display import HTML, display
from base64 import b64encode
from functools import wraps
from threading import Thread
from ipywidgets import Output
from collections import deque
import multiprocessing
import subprocess
import ctypes
import queue
import time
import re
import os

def show_video(fname, dimensions=(288, 512), fmt='webm', verbose=True):
    mp4 = open(fname,'rb').read()
    data_url = "data:video/{};base64,".format(fmt) + b64encode(mp4).decode() 
    
    html = (
      '<video width="{width:}" height="{height:}" controls autoplay>\n' +
      '  <source src="{data_url:}" type="video/{fmt:}">\n'        +
      '  Your browser does not support the video tag.\n'       +
      '</video>'
    ).format(
        width=dimensions[0],
        height=dimensions[1],
        data_url=data_url,
        fmt=fmt
    )

    if verbose:
        print(fname, flush=True)

    display(HTML(html))

class SegmentMonitor(Thread):
    def __init__(self, fname, segment_fmt, callback, current_segment, sleep_interval=1):
        super().__init__()
        self.fname = fname
        self.segment_fmt = segment_fmt
        self.callback = callback
        self.current_segment = current_segment
        self.next_fname = self.fname.format(self.segment_fmt.format(self.current_segment[0]+1))
        self.sleep_interval = sleep_interval
        self.toStop = False

        # remove segment video files for the next two steps if they already
        # exist (otherwise their existence will confuse the SegmentMonitor)
        self.clean_files()

    def clean_files(self):
        dirname = os.path.dirname(self.fname)
        fname_re = re.compile(re.escape(
            os.path.basename(self.fname)
        ).replace('\\{', '{').replace('\\}', '}').format('[0-9]+'))

        for file in os.listdir(dirname):
            if not fname_re.fullmatch(file) is None:
                os.remove(os.path.join(dirname,file))

    def stop(self):
        self.toStop = True

    def run(self):
        while not self.toStop:
            if os.path.isfile(self.next_fname):
                self.callback(self.fname.format(self.segment_fmt.format(self.current_segment[0])))
                self.current_segment[0] += 1
                self.next_fname = self.fname.format(self.segment_fmt.format(self.current_segment[0]+1))

            time.sleep(self.sleep_interval)

class ScreenRecorder(ContextDecorator):
    def __init__(self, display, display_size, video_path,
                 video_basename="vid", segment_time=None,
                 video_callback=show_video, vid_counter=None):
        super().__init__()
        self.vid_counter = [0] if vid_counter is None else vid_counter
        self.display = display
        self.display_size = display_size
        self.video_path = video_path
        self.video_basename = video_basename
        self.segment_time = segment_time
        self.video_callback = video_callback
        self.current_segment = [0]

        if self.segment_time is None:
            self.segment_cmd = ''
            self.filename_segment = ''
            self.filename_segment_py = ''
        else:
            self.segment_cmd = '-f segment -segment_time {} -reset_timestamps 1'.format(self.segment_time)
            self.filename_segment = '%03d'
            self.filename_segment_py = '{:03d}'

        os.makedirs(video_path, exist_ok=True)
  
    def start(self):
        self.fname = os.path.join(self.video_path, "{}_{}{}.webm".format(
            self.video_basename, self.vid_counter[0], self.filename_segment))
        self.py_fname = os.path.join(self.video_path, "{}_{}{}.webm".format(
            self.video_basename, self.vid_counter[0], '{}'))
        self.vid_counter[0] += 1

        self.current_segment[0] = 0

        if self.segment_time is None:
            self.segment_thread = None
        else:
            self.segment_thread = SegmentMonitor(self.py_fname,
                self.filename_segment_py, self.video_callback,
                self.current_segment, sleep_interval=self.segment_time/5)
            self.segment_thread.start()

        cmd = ('ffmpeg -y -r 30 -f x11grab -draw_mouse 0 -s {}x{} -i :{} -c:v libvpx ' +
               '-quality realtime -cpu-used 0 -b:v 384k -qmin 10 -qmax 42 -maxrate 384k ' +
               '-bufsize 100k -an {} {}'
        ).format(
            self.display_size[0], self.display_size[1],
            self.display.display, self.segment_cmd, self.fname
        )
        
        self.ffmpegHandle=subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)
      
    def stop(self):
        self.ffmpegHandle.communicate(b"q")

        if not self.segment_thread is None:
            self.segment_thread.toStop = True
            self.segment_thread.join()

        if not self.video_callback is None:
            self.video_callback(self.py_fname.format(
                self.filename_segment_py.format(self.current_segment[0])),
                dimensions=self.display_size
            )
    
    def __enter__(self, *args, **kwargs):
        self.start(*args, **kwargs)
        
    def __exit__(self, *exc):
        self.stop()

class GuiThread(Thread):
    def __init__(self, max_gui_outputs=10):
        super().__init__()
        self.callback_queue = multiprocessing.JoinableQueue()
        self.sync_event = multiprocessing.Event()
        self.toStop = False
        self.outputs = deque(maxlen=max_gui_outputs)

    def stop(self):
        self.toStop = True
        self.callback_queue.put(None)
        self.join()

    def run(self):
        try:
            while True:
                if self.toStop: # empty the queue, the exit
                    c = self.callback_queue.get_nowait()
                    self.callback_queue.task_done()
                else:
                    c = self.callback_queue.get()
                    if c is None:
                        self.callback_queue.task_done()
                    else:
                        c, args, kwargs = c
                        try:
                            out = Output()
                            display(out)
                            
                            with out:
                                c(*args, **kwargs)

                            if len(self.outputs) >= self.outputs.maxlen:
                                self.outputs[0].close()

                            self.outputs.append(out)
                            self.callback_queue.task_done()
                        except:
                            self.callback_queue.task_done()
                            raise

        except queue.Empty:
            pass

class GuiCallbackWrapper:
    """
    A wrapper for a callback that manages its execution
    inside the GuiThread.
    """
    def __init__(self, callback, callback_queue):
        self.callback = callback
        self.callback_queue = callback_queue

    def __call__(self, *args, **kwargs):
        return self.callback_queue.put((self.callback, args, kwargs))

class WPExitHandler:
    def __init__(self, process, events):
        self.process = process
        self.events = events

    def __call__(self):
        self.process.join()
        for event in self.events:
            event.set()

class WorkerProcess:
    """
    A class that helps to run stuff inside a separate process
    to prevent the main process from crashing when stuff
    goes wrong.
    """    
    def __init__(self, initialize_func, *args, num_retries=3,
                 arg_processor=None, **kwargs):
        """
        arg_processor: A function which, before calling func in run_func, gets
                      func's *args, **kwargs as input arguments and returns
                      their processed version which are then ready to be passed
                      to func. 
        """
        self.doneEvent = multiprocessing.Event()
        self.function_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.initialize_func = initialize_func
        self.num_retries = num_retries
        self.args = args
        self.kwargs = kwargs
        self.arg_processor = arg_processor
        self.process = self.create_process()
 
    def create_process(self):
        return multiprocessing.Process(
            target=self._run,
            args=(self.initialize_func, self.args, self.kwargs,
                  self.function_queue, self.result_queue, self.doneEvent)
        )
    
    def start(self):
        self.doneEvent.clear()
        self.process.start()
        self.doneEvent.wait()
        self.doneEvent.clear()

        self.join_thread = Thread(
            target=WPExitHandler(self.process, [self.doneEvent]))
        self.join_thread.start()
            
    @staticmethod
    def _run(initialize_func, args, kwargs, function_queue, result_queue, doneEvent):
        init_ret = initialize_func(*args, **kwargs)
        doneEvent.set()
        
        while True:
            item = function_queue.get()
            if item is None: break
            func, args, kwargs = item
            ret = func(init_ret, *args, **kwargs)
            result_queue.put(ret)
            doneEvent.set()
            
    def reinit(self):
        self.process.terminate()

        # clear function_queue
        try:
            while True:
                self.function_queue.get_nowait()
        except multiprocessing.queues.Empty:
            pass
        
        # clear result_queue
        try:
            while True:
                self.result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            pass
                    
        self.doneEvent.clear()
        self.process = self.create_process()
        self.start()

    def run_func(self, func, *args, **kwargs):
        """
        Runs the specified function in the process.
        """
        # reinitialize the process if it died
        if not self.process.is_alive():
            self.reinit()

        if not self.arg_processor is None:
            args, kwargs = self.arg_processor(args, kwargs)
        
        for i in range(self.num_retries+1):
            if i > 0:
                print("Execution crashed. Retrying {}/{}...".format(i,
                    self.num_retries), flush=True)

            self.doneEvent.clear()
            self.function_queue.put((func, args, kwargs))
            self.doneEvent.wait()
            
            try:
                res = self.result_queue.get_nowait()
                return res
            except:
                self.reinit()
                
        raise RuntimeError("Execution crashed on all {} tries.".format(
            self.num_retries))

    def __call__(self, func):
        """
        When called on a function, returns a decorated version that
        uses run_func internally.
        """
        orig_name = func.__name__
        orig_qualname = func.__qualname__
        func.__name__ = func.__name__ + "_inner__"
        func.__qualname__ = func.__qualname__ + "_inner__"
        
        current_module = __import__('__main__')
        setattr(current_module, func.__name__, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.run_func(func, *args, **kwargs)

        wrapper.__name__ = orig_name
        wrapper.__qualname__ = orig_qualname
        setattr(current_module, wrapper.__name__, wrapper)
        
        if not self.process.is_alive():
            self.start()

        return wrapper

class ScreenRecordWrapper:
    def __init__(self, func):
        self.func = func
        
    def __call__(self, screen_recorder, *args, **kwargs):
        with screen_recorder:
            ret = self.func(*args, **kwargs)
        
        return ret

class ScreenCastProcess(WorkerProcess):
    """
    A WorkerProcess that will additionally create a virtual
    display and record any GUI output that the tasks produce.
    """
    def __init__(self, gui_init_func=lambda *args, **kwargs: None,
                 display_size=(288, 512), show_videos=True,
                 video_path="output", num_retries=3, arg_processor=None,
                 segment_time=None, show_video_func=show_video,
                 max_gui_outputs=10):
        def initialize_func(display_size, video_path, vid_counter, callback_queue):
            # to decrement counter on a retry after a crash
            vid_counter[0] -= 1
            
            global screen_recorder
            from pyvirtualdisplay import Display
            import copy
            import os
            
            DISPLAY = Display(visible=0, size=display_size)
            disp_ret = DISPLAY.start()
            os.environ["DISPLAY"] = ":{}".format(DISPLAY.display)

            screen_recorder = ScreenRecorder(
            DISPLAY, display_size, video_path, 
            segment_time=segment_time,
            video_callback=GuiCallbackWrapper(show_video_func, callback_queue)
                              if show_videos else None,
            vid_counter=vid_counter)

            gui_init_func(display_size, video_path, vid_counter,
                          callback_queue, screen_recorder)

            return screen_recorder

        self.gui_thread = GuiThread(max_gui_outputs=max_gui_outputs)
        self.gui_thread.start()
        self.vid_counter = multiprocessing.Array(ctypes.c_long, 1)
        self.vid_counter[0] = 1 # will be decremented back to 0 in init

        super().__init__(initialize_func,
                         display_size=display_size,
                         video_path=video_path,
                         vid_counter=self.vid_counter,
                         callback_queue=self.gui_thread.callback_queue,
                         num_retries=num_retries,
                         arg_processor=arg_processor)

    def run_func(self, func, *args, **kwargs):
        ret = super().run_func(ScreenRecordWrapper(func), *args, **kwargs)
        self.gui_thread.callback_queue.join()
        return ret

    def __del__(self):
        self.gui_thread.stop()
