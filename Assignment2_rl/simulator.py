import sys
import json
from Classes import SimBuffer, NetworkTrace, Scorecard, simulator_comm

#this file was written by Zach Peats
#This is the video download and playback simulator for an ABR algorithm lab.
#The program simulates a video stream over a network, using a network trace and a video manifest
#for more information regarding usage and file specifications, check out the readme



verbose = False


def loadtrace(tracefile):

    with open(tracefile, 'r',encoding='utf-8') as infile:
        lines = infile.readlines()

    tracelog = []

    for line in lines:
        splitline = line.split(' ')
        if len(splitline) > 1:
            try:
                tracelog.append((float(splitline[0]), float(splitline[1])))

            except ValueError as e:
                print("Your trace file is poorly formed!")



    trace = NetworkTrace.NetworkTrace(tracelog)

    return trace


def loadmanifest(manifestfile):

    with open(manifestfile, 'r', encoding='utf-8') as infile:
        lines = infile.read()

    manifest = json.loads(lines)
    return manifest

def prep_bitrates(available_rates, chunk):
    rates = dict(map(lambda x, y: (x, y), available_rates, chunk))
    return rates

def prep_chunk(chunks_rem, manifest, chunk_num):
    params = {  "left" : chunks_remaining,
                "time" : manifest["Chunk_Time"],
                "current" : chunk_num
                }
    return params


# if __name__ == "__main__":
class official_sim:
    def __init__(self, trace_file, manifest_file):
        #check arguments for relevant flags

        # if "-v" in sys.argv or "--verbose" in sys.argv:
        #     verbose = True
        verbose = False


        #Load in network trace from input file

        # trace = loadtrace(sys.argv[1])
        self.trace = loadtrace(trace_file)

        #read video manifest

        # manifest = loadmanifest(sys.argv[2])
        self.manifest = loadmanifest(manifest_file)

        #create scorecard for logging

        self.logger = Scorecard.Scorecard(1, 1, 1)

        #simulator setup

        self.buffer = SimBuffer.SimBuffer(manifest["Buffer_Size"])

        self.chunks_remaining = manifest["Chunk_Count"]
        self.current_time = 0
        self.prev_throughput = 0
        self.rebuff_time = 0
        self.pref_bitrate = manifest["Preferred_Bitrate"]

        self.stu_chunk_size = None

        self.chunk_list = [(key, value) for key, value in manifest["Chunks"].items()]

        self.chunk_iter = chunk_list.__iter__()

        #Communication loop with student (for all chunks):

        self.chunknum, self.chunk = next(chunk_iter, None)

    def official_sim_loop(self):
        #calculate and pack info to be sent to student
        # todo ensure input types are correct
        m_band = self.trace.get_current_timesegment(self.current_time)[1]
        buf_occ = self.buffer.get_student_params()
        av_bitrates = prep_bitrates(self.manifest["Available_Bitrates"], self.chunk)
        chunk_arg = prep_chunk(self.chunks_remaining, self.manifest, self.chunknum)





        #send info to student, get response
        chosen_bitrate = simulator_comm.send_req_json(m_band, prev_throughput, buf_occ, av_bitrates, current_time, chunk_arg, rebuff_time, pref_bitrate)


        #bad response checking, ensure chunk fits in buffer
        try:
            self.stu_chunk_size = av_bitrates[int(chosen_bitrate)]
        except( KeyError ):
            print("Student returned invalid bitrate, exiting")
            break

        if self.stu_chunk_size > self.buffer.available_space():
            #chunk chosen does not fit in buffer, wait .5s and resend request
            buffer_time = self.buffer.burn_time(.5)
            self.current_time += .5
            continue


        self.logger.log_bitrate_choice(self.current_time, self.chunknum, (chosen_bitrate, self.stu_chunk_size))

        #simulate download and playback
        time_elapsed = self.trace.simulate_download_from_time(self.current_time, self.stu_chunk_size)

        #round time to remove floating point errors
        #todo: this did not fix them
        time_elapsed = round(time_elapsed, 3)

        self.rebuff_time = self.buffer.sim_chunk_download(self.stu_chunk_size, chunk_arg["time"], time_elapsed)

        #update state variables
        prev_throughput = (stu_chunk_size * 8) / time_elapsed
        self.current_time += time_elapsed
        self.chunks_remaining -= 1

        self.logger.log_rebuffer(self.current_time - self.rebuff_time, self.rebuff_time)

        #log actions



        #get next chunk
        chunknum, chunk = next(chunk_iter, (None, None))