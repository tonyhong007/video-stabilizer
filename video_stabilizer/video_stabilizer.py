# import cv2
# import time
# import grpc
# import unary.unary_pb2_grpc as pb2_grpc
# import unary.unary_pb2 as pb2
# import numpy as np
# from collections import defaultdict

"""
class Sink(object):

    def set_expected_frames(self, video_index, num_frames):
        self.num_frames_left[video_index] = num_frames
        print("Expecting", self.num_frames_left[video_index], "total frames from video", video_index)

    def send(self, frame_index, transform, timestamp):
        with ray.profiling.profile("Sink.send"):
            if frame_index < len(self.latencies[video_index]):
                return
            assert frame_index == len(self.latencies[video_index]), frame_index

            self.latencies[video_index].append(time.time() - timestamp)

            self.num_frames_left[video_index] -= 1
            if self.num_frames_left[video_index] % 100 == 0:
                print("Expecting", self.num_frames_left[video_index], "more frames from video", video_index)

            if self.num_frames_left[video_index] == 0:
                print("Video {} DONE".format(video_index))
                if self.last_view is not None:
                    ray.get(self.last_view)
                self.signal.send.remote()

            if self.viewer is not None and video_index == 0:
                self.last_view = self.viewer.send.remote(transform)

            if self.checkpoint_interval != 0 and frame_index % self.checkpoint_interval == 0:
                ray.experimental.internal_kv._internal_kv_put(video_index, frame_index, overwrite=True)

    def latencies(self):
        latencies = []
        for video in self.latencies.values():
            for i, l in enumerate(video):
                latencies.append((i, l))
        return latencies

class Decoder:
    def __init__(self, filename, start_frame):
        self.v = cv2.VideoCapture(filename)
        self.v.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def decode(self, frame):
        if frame != self.v.get(cv2.CAP_PROP_POS_FRAMES):
            print("next frame", frame, ", at frame", self.v.get(cv2.CAP_PROP_POS_FRAMES))
            self.v.set(cv2.CAP_PROP_POS_FRAMES, frame)
        grabbed, frame = self.v.read()
        assert grabbed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        return frame

    def ready(self):
        return

class StabilizeClient(object):
    # Client for gRPC functionality

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.UnaryStub(self.channel)

    def get_stabilized_frame_image(self, frame_image, prev_frame, features, trajectory, padding, transforms, frame_index):
        # Client function to call the rpc for StabilizeRequest

        frame_image_request = pb2.StabilizeRequest(frame_image=frame_image, prev_frame=prev_frame, features=features, trajectory=trajectory, padding=padding, transforms=transforms, frame_index=frame_index)
        return self.stub.GetStabilizeResponse(frame_image_request)

def process_videos(video_pathname, num_videos, output_filename):
    # Initializing a sink
    sink = Sink()

    v = cv2.VideoCapture(video_pathname)
    num_total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(v.get(cv2.CAP_PROP_FPS))
    print("Processing total frames", num_total_frames, "from video", video_pathname)
    for i in range(num_videos):
        sink.set_expected_frames(i, num_total_frames - 1)

    decoder = Decoder(video_pathname, 0)
    start_frame = 0
    radius = fps

    # Start at `radius` before the start frame, since we need to compute a
    # moving average.
    next_to_send = start_frame
    start_frame -= radius
    if start_frame < 0:
        padding = start_frame * -1
        start_frame = 0
    else:
        padding = 0

    frame_timestamps = []
    trajectory = []
    transforms = []

    # 3D array
    features = []

    frame_timestamp = start_frame / fps
    diff = frame_timestamp - time.time()
    if diff > 0:
        time.sleep(diff)
    frame_timestamps.append(frame_timestamp)
    prev_frame = decoder.decode.remote(start_frame)

    stabilize_client = StabilizeClient()
    for frame_index in range(start_frame, num_total_frames - 1):
        frame_timestamp = (start_frame + frame_index + 1) / fps
        diff = frame_timestamp - time.time()
        if diff > 0:
            time.sleep(diff)
        frame_timestamps.append(frame_timestamp)

        frame = decoder.decode.remote(start_frame + frame_index + 1)

        result = stabilize_client.get_stabilized_frame_image(frame, prev_frame, features, trajectory, padding, transforms, frame_index)

        prev_frame = result.stabilized_frame_image
        features = result.features
        trajectory = result.trajectory
        transforms = result.transforms

    # TODO: Should we be calling smooth here?
    while next_to_send < num_total_frames - 1:
        trajectory.append(trajectory[-1])
        midpoint = radius
        final_transform = smooth.options(resources={
            resource: 0.001
            }).remote(transforms.pop(0), trajectory[midpoint], *trajectory)
        trajectory.pop(0)

        final = sink.send.remote(next_to_send, final_transform,
                frame_timestamps.pop(0))
        next_to_send += 1


    # Wait for all video frames to complete
    ready = 0
    while ready != num_videos:
        time.sleep(1)
        ready = ray.get(signal.wait.remote())

    latencies = []
    for sink in sinks:
        latencies += ray.get(sink.latencies.remote())
    if output_filename:
        with open(output_filename, 'w') as f:
            for t, l in latencies:
                f.write("{} {}\n".format(t, l))
    else:
        for latency in latencies:
            print(latency)
    latencies = [l for _, l in latencies]
    print("Mean latency:", np.mean(latencies))
    print("Max latency:", np.max(latencies))
"""

def main(args):
    print("hello world")
    #process_videos(args.video_path, args.num_videos, args.output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the video stabilizer.")

    parser.add_argument("--num-videos", required=True, type=int)
    parser.add_argument("--video-path", required=True, type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    main(args)