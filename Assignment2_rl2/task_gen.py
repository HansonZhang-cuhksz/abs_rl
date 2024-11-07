import json
import random

def gen_manifest():
    # Unchanged parameters
    video_time = 60
    chunk_count = 30
    chunk_time = 2
    available_bitrates_1 = [
        50000,
        100000,
        500000
    ]
    available_bitrates_2 = [
            500000,
            1000000,
            5000000
    ]
    preferred_bitrate = random.choice([None,"5000000"])
    available_bitrates = random.choice([available_bitrates_1,available_bitrates_2])
    # print(available_bitrates)
    task = ""
    # generate chunks
    chunks = {}
    for i in range(chunk_count):
        if available_bitrates[0] <= 50000: #task1
            buffer_size = 40000000
            chunk_bitrate = random.randint(12000,13000)  # Random size for bitrate 50000
            task = "testHD" if preferred_bitrate == None else random.choice(["badtest","testALThard","testALTsoft"])
        else:
            buffer_size = 200000
            chunk_bitrate = random.randint(120000,130000)  # Random size for bitrate 500000
            task = "testPQ" if preferred_bitrate == None else "testHDmanPQtrace"
        chunks[str(i)] = [
            chunk_bitrate,  # Random size for bitrate 500000
            random.randint(245000, 255000),  # Random size for bitrate 1000000
            random.randint(1230000, 1280000) # Random size for bitrate 5000000
        ]
    print(task)


    # Create the dictionary
    data = {
        "Video_Time": video_time,
        "Chunk_Count": chunk_count,
        "Chunk_Time": chunk_time,
        "Buffer_Size": buffer_size,
        "Available_Bitrates": available_bitrates,
        "Preferred_Bitrate": None,
        "Chunks": chunks
    }
    # print(data)


    # Write data to a JSON file
    f = open("manifest.json","w")
    json.dump(data,f,indent = 4)

    manifest_file = f"./manifest.json"
    return manifest_file

def gen_trace():
    trace_file_number = random.randint(1, 6)
    trace_file = f"./traceFiles/trace_{trace_file_number}.txt"
    return trace_file