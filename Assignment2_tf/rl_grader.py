import os
import time
import subprocess
import threading
# import sys

#This file was written by Zach Peats
#This program repeatedly invokes Simulator.py and studentComm.py in order to run multiple testcases autonomously
#For more information check out the readme



def run_student_code():
    subprocess.run(['python', 'studentComm.py'])
    return


# if __name__ == "__main__":
def grade():
    #check for verbosity
    verboseflag = ""
    # if "-v" in sys.argv or "--verbose" in sys.argv:
    #     verboseflag = "-v"

    switch_ratio = 1
    buffer_ratio = 1



    manifestfilename = 'manifest.json'
    tracefilename = 'trace.txt'



    # outtext = []
    scores = []

    #iterate over all test directories
    for testdir in os.listdir('./tests/'):
        testpath = "./tests/" + testdir + "/"

        manifestpath = testpath + manifestfilename
        tracepath = testpath + tracefilename

        #Run student process
        student_thread = threading.Thread(target=run_student_code)
        student_thread.start()
        time.sleep(1)

        #run simulator process
        output = subprocess.run(['python', 'simulator.py', tracepath, manifestpath, verboseflag], capture_output=True )
        # print(output)

        student_thread.join()

        #decode output and come up with a final score

        outputlines = output.stdout.decode('unicode_escape').split('\n')

        sanitizedoutput = [line.strip() for line in outputlines]

        # outtext.append(testdir + ": ")
        # outtext.append("\n")
        # outtext += outputlines
        # outtext.append("\n")

        average_bitrate = None
        buffer_time = None
        switches = None

        for line in sanitizedoutput:
            if "Average bitrate" in line:
                average_bitrate = float(line.split(':')[1])

            if "buffer time" in line:
                buffer_time = float(line.split(':')[1])

            if "switches" in line:
                switches = float(line.split(':')[1])

        #check to ensure that diagnostic scores are found
        if switches is not None and buffer_time is not None and average_bitrate is not None:
            buffer_penalty = pow( (1 - (.05 * buffer_ratio)), buffer_time)
            switch_penalty = pow( (1 - (.08 * switch_ratio)), switches)

            scores.append(average_bitrate * buffer_penalty * switch_penalty)

        else:
            raise Exception("Unexpected output from the simulator")
        
    # badtest_multiplier = 132.9976199980265
    # testALThard_multiplier = 117.1617545262574
    # testALTsoft_multiplier = 2.451864142941845
    # testHD_multiplier = 0.2614116056256966
    # testHDmanPQtrace = 5069054.146379368
    # testPQ_multiplier = 6159002.792392146
    # score = scores[0] * badtest_multiplier + scores[1] * testALThard_multiplier + scores[2] * testALTsoft_multiplier + scores[3] * testHD_multiplier + scores[4] * testHDmanPQtrace + scores[5] * testPQ_multiplier
    # score /= 6
    # return score

        #     outtext.append("Score:")
        #     outtext.append(str(score))
        #     outtext.append('\n')
        #     outtext.append('\n')
        #     outtext.append('\n')

        # else:
        #     outtext.append("Unexpected output from the simulator")
        #     outtext.append('\n')
        #     outtext.append('\n')


    # with open("grade.txt", 'w', encoding='utf-8') as outfile:
    #     outfile.writelines(outtext)

if __name__ == "__main__":
    print(grade())