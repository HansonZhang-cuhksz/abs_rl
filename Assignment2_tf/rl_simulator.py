import sys
import json
from Classes import SimBuffer, NetworkTrace, Scorecard#, simulator_comm

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

def sim_grade(measured_bandwidth, selected_bitrate):
    # Simulate downloading the chunk based on the selected bitrate
    if measured_bandwidth < selected_bitrate:
        reward = -1  # Penalty for rebuffering
    else:
        reward = 1  # Reward for successful playback
    return reward

# if __name__ == "__main__":
def init(trace_pth, manifest_pth):
    global verbose, trace, manifest, logger, buffer, chunks_remaining, current_time, prev_throughput, rebuff_time, pref_bitrate, stu_chunk_size, chunk_list, chunk_iter
    global chunknum, chunk

    #check arguments for relevant flags

    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True


    #Load in network trace from input file

    # trace = loadtrace(sys.argv[1])
    trace = loadtrace(trace_pth)

    #read video manifest

    # manifest = loadmanifest(sys.argv[2])
    manifest = loadmanifest(manifest_pth)

    #create scorecard for logging

    logger = Scorecard.Scorecard(1, 1, 1)

    #simulator setup

    buffer = SimBuffer.SimBuffer(manifest["Buffer_Size"])

    chunks_remaining = manifest["Chunk_Count"]
    current_time = 0
    prev_throughput = 0
    rebuff_time = 0
    pref_bitrate = manifest["Preferred_Bitrate"]

    stu_chunk_size = None

    chunk_list = [(key, value) for key, value in manifest["Chunks"].items()]

    chunk_iter = chunk_list.__iter__()

    #Communication loop with student (for all chunks):

    chunknum, chunk = next(chunk_iter, None)

def loop(agent):
    global verbose, trace, manifest, logger, buffer, chunks_remaining, current_time, prev_throughput, rebuff_time, pref_bitrate, stu_chunk_size, chunk_list, chunk_iter
    global chunknum, chunk

    # while chunk:
    if 1:
        #calculate and pack info to be sent to student
        # todo ensure input types are correct
        m_band = trace.get_current_timesegment(current_time)[1]
        buf_occ = buffer.get_student_params()
        av_bitrates = prep_bitrates(manifest["Available_Bitrates"],chunk)
        chunk_arg = prep_chunk(chunks_remaining, manifest, chunknum)





        #send info to student, get response
        #chosen_bitrate = simulator_comm.send_req_json(m_band, prev_throughput, buf_occ, av_bitrates, current_time, chunk_arg, rebuff_time, pref_bitrate)
        state = [m_band, prev_throughput, buf_occ["size"], buf_occ["current"], buf_occ["time"], current_time, rebuff_time]# + [int(btr) for btr in sorted(av_bitrates.keys())]
        action = agent.act(state)
        chosen_bitrate = sorted(av_bitrates.keys())[action]

        #bad response checking, ensure chunk fits in buffer
        try:
            stu_chunk_size = av_bitrates[int(chosen_bitrate)]
        except( KeyError ):
            print("Student returned invalid bitrate, exiting")
            # break
            return

        if stu_chunk_size > buffer.available_space():
            #chunk chosen does not fit in buffer, wait .5s and resend request
            buffer_time = buffer.burn_time(.5)
            current_time += .5
            # continue
            # return


        logger.log_bitrate_choice(current_time, chunknum, (chosen_bitrate, stu_chunk_size))

        #simulate download and playback
        time_elapsed = trace.simulate_download_from_time(current_time, stu_chunk_size)

        #round time to remove floating point errors
        #todo: this did not fix them
        time_elapsed = round(time_elapsed, 3)

        rebuff_time = buffer.sim_chunk_download(stu_chunk_size, chunk_arg["time"], time_elapsed)

        #update state variables
        prev_throughput = (stu_chunk_size * 8) / time_elapsed
        current_time += time_elapsed
        chunks_remaining -= 1

        logger.log_rebuffer(current_time - rebuff_time, rebuff_time)

        #log actions

        #get next chunk
        chunknum, chunk = next(chunk_iter, (None, None))

        m_band = trace.get_current_timesegment(current_time)[1]
        buf_occ = buffer.get_student_params()
        av_bitrates = prep_bitrates(manifest["Available_Bitrates"],chunk)
        chunk_arg = prep_chunk(chunks_remaining, manifest, chunknum)
        next_state = [m_band, prev_throughput, buf_occ["size"], buf_occ["current"], buf_occ["time"], current_time, rebuff_time]

        reward = sim_grade(m_band, chosen_bitrate) + 3
        if rebuff_time == False:
            reward = -3

        done = chunks_remaining == 1

        return action, reward, next_state, state, done


    # #cleanup and return
    # simulator_comm.send_exit()

    # if(verbose):
    #     logger.output_verbose()
    # else:
    #     logger.output_results()