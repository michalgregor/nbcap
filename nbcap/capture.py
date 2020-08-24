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

def show_video(fname, dimensions=(800, 600), fmt='webm',
               verbose=True, remove_after=True):
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
    if remove_after:
        os.remove(fname)

class ShowVideoCallback:
    def __init__(self, dimensions, fmt='webm', verbose=True, remove_after=True):
        self.dimensions = dimensions
        self.fmt = fmt
        self.verbose = verbose
        self.remove_after = remove_after
    
    def __call__(self, fname):
        return show_video(fname, dimensions=self.dimensions,
                          fmt=self.fmt, verbose=self.verbose,
                          remove_after=self.remove_after)

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
            self.display, self.segment_cmd, self.fname
        )
        
        self.ffmpegHandle=subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)
      
    def stop(self):
        self.ffmpegHandle.communicate(b"q")

        if not self.segment_thread is None:
            self.segment_thread.toStop = True
            self.segment_thread.join()

        if not self.video_callback is None:
            self.video_callback(self.py_fname.format(
                self.filename_segment_py.format(self.current_segment[0])))
    
    def __enter__(self, *args, **kwargs):
        self.start(*args, **kwargs)
        
    def __exit__(self, *exc):
        self.stop()

class OutputWrapper:
    def __init__(self, callback, outputs):
        self.callback = callback
        self.outputs = outputs
    
    def __call__(self, *args, **kwargs):
        ret = None
        out = Output()
        display(out)
        
        with out:
            ret = self.callback(*args, **kwargs)

        if len(self.outputs) >= self.outputs.maxlen:
            self.outputs[0].close()

        self.outputs.append(out)
        
        return ret

class OutputManager:
    def __init__(self, *args, max_gui_outputs=10, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.outputs = deque(maxlen=max_gui_outputs)

    def __call__(self, callback):
        return OutputWrapper(callback, self.outputs, *self.args, **self.kwargs)

class GuiThread(Thread):
    def __init__(self, output_manager=None):
        super().__init__()
        self.output_manager = output_manager
        self.callback_queue = multiprocessing.JoinableQueue()
        self.sync_event = multiprocessing.Event()
        self.toStop = False
        self.toClear = False

    def stop(self):
        self.toStop = True
        self.callback_queue.put(None)
        self.join()

    def clear(self):
        self.sync_event.clear()
        self.toClear = True
        self.callback_queue.put(None)
        self.sync_event.wait()

    def run(self):
        try:
            while True:
                if self.toStop: # empty the queue, the exit
                    c = self.callback_queue.get_nowait()
                    self.callback_queue.task_done()
                elif self.toClear:
                    try:
                        c = self.callback_queue.get_nowait()
                        self.callback_queue.task_done()
                    except multiprocessing.queues.Empty:
                        self.toClear = False
                        self.sync_event.set()
                else:
                    c = self.callback_queue.get()
                    if c is None:
                        self.callback_queue.task_done()
                    else:
                        c, args, kwargs = c
                        if not self.output_manager is None:
                            c = self.output_manager(c)
                        try:
                            c(*args, **kwargs)
                            self.callback_queue.task_done()
                        except:
                            self.callback_queue.task_done()
                            raise

        except multiprocessing.queues.Empty:
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
        self.manager = multiprocessing.Manager()
        self.doneEvent = multiprocessing.Event()
        self.function_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
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

    def _clear(self):
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

    def reinit(self):
        if not self.process is None and self.process.is_alive():
            self.process.terminate()
        
        self._clear()
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
            except multiprocessing.queues.Empty:
                self.reinit()
                
        raise RuntimeError("Execution crashed on all {} tries.".format(
            self.num_retries))

    def __call__(self, func):
        """
        When called on a function, returns a decorated version that
        uses run_func internally.
        """
        # orig_name = func.__name__
        # orig_qualname = func.__qualname__
        # func.__name__ = func.__name__ + "_inner__"
        # func.__qualname__ = func.__qualname__ + "_inner__"
        
        # current_module = __import__('__main__')
        # setattr(current_module, func.__name__, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.run_func(func, *args, **kwargs)

        # wrapper.__name__ = orig_name
        # wrapper.__qualname__ = orig_qualname
        # setattr(current_module, wrapper.__name__, wrapper)
        
        if not self.process.is_alive():
            self.start()

        return wrapper

class ScreenRecordWrapper:
    def __init__(self, func):
        self.func = func
        
    def __call__(self, init_ret, *args, **kwargs):
        screen_recorder = init_ret

        with screen_recorder:
            ret = self.func(*args, **kwargs)
        
        return ret

class DisplayProcess:
    def __init__(self, display_size):
        self.display_size = display_size       
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self._run, args=(display_size, self.queue, self.stop_event)
        )
        
        self.process.start()
        self.id = self.queue.get()        
        os.environ["DISPLAY"] = ":{}".format(self.id)

    @staticmethod
    def _run(display_size, queue, stop_event):
        import pyvirtualdisplay
        import os
        os.setpgrp()

        pydisp = pyvirtualdisplay.Display(visible=0, size=display_size)
        display = pydisp.start().display
        queue.put(display)
        stop_event.wait()
        pydisp.stop()
        queue.put(None)

    def stop(self):
        self.stop_event.set()
        self.queue.get()

    def __del__(self):
        self.stop()

class ScreenCastProcess(WorkerProcess):
    """
    A WorkerProcess that will additionally create a virtual
    display and record any GUI output that the tasks produce.
    """
    def __init__(self, gui_init_func=lambda *args, **kwargs: None,
                 display_size=(288, 512), show_videos=True,
                 video_path="output", num_retries=3, arg_processor=None,
                 segment_time=None, show_video_func=show_video,
                 output_manager="default"):
        def initialize_func(display_size, video_path, vid_counter, callback_queue):
            # to decrement counter on a retry after a crash
            vid_counter[0] -= 1
            
            from pyvirtualdisplay import Display
            import copy
            import os
            
            DISPLAY = Display(visible=0, size=display_size)
            disp_ret = DISPLAY.start()
            os.environ["DISPLAY"] = ":{}".format(DISPLAY.display)

            screen_recorder = ScreenRecorder(
                DISPLAY.display, display_size, video_path, 
                segment_time=segment_time,
                video_callback=GuiCallbackWrapper(show_video_func, callback_queue)
                        if show_videos else None,
                vid_counter=vid_counter
            )

            gui_init_func(display_size, video_path, vid_counter,
                          callback_queue, screen_recorder)

            return screen_recorder

        if isinstance(output_manager, str) and output_manager == "default":
            output_manager = OutputManager()

        self.gui_thread = GuiThread(output_manager=output_manager)
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

    def _clear(self):
        super()._clear()
        # clear gui thread
        self.gui_thread.clear()

    def run_func(self, func, *args, **kwargs):
        ret = super().run_func(ScreenRecordWrapper(func), *args, **kwargs)
        self.gui_thread.callback_queue.join()
        return ret

    def __del__(self):
        if hasattr(self, 'gui_thread'):
            self.gui_thread.stop()
