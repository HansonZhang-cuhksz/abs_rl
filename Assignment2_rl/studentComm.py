import socket
import json
import studentcode_123090823 as studentcode

# Quick-and-dirty TCP Server:
# ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ss.bind(('localhost', 6000))
ss.listen(10)
# print('Waiting for simulator')

(clientsocket, address) = ss.accept()


def recv_commands():
    message = ""

    # data = dict()

    while(1):
        messagepart = clientsocket.recv(2048).decode()

        message += messagepart
        if message[-1] == '\n':


            jsonargs = json.loads(messagepart)
            message = ""

            if(jsonargs["exit"] != 0):
                break

            #todo: json data sanitization
            # Measured_Bandwidth = jsonargs["Measured Bandwidth"]
            # Previous_Throughput = jsonargs["Previous Throughput"]
            # Buffer_Occupancy = jsonargs["Buffer Occupancy"]
            # Video_Time = jsonargs["Video Time"]
            # Chunk = jsonargs["Chunk"]
            # Rebuffering_Time = jsonargs["Rebuffering Time"]
            # Preferred_Bitrate = jsonargs["Preferred Bitrate"]

            bitrate = studentcode.student_entrypoint(jsonargs["Measured Bandwidth"], jsonargs["Previous Throughput"], jsonargs["Buffer Occupancy"], jsonargs["Available Bitrates"], jsonargs["Video Time"], jsonargs["Chunk"], jsonargs["Rebuffering Time"], jsonargs["Preferred Bitrate"])
            
            # # To file
            # data[Video_Time] = {
            #     "Measured_Bandwidth": Measured_Bandwidth,
            #     "Previous_Throughput": Previous_Throughput,
            #     "Buffer_Occupancy": Buffer_Occupancy,
            #     "Video_Time": Video_Time,
            #     "Chunk": Chunk,
            #     "Rebuffering_Time": Rebuffering_Time,
            #     "Preferred_Bitrate": Preferred_Bitrate
            # }

            payload = json.dumps({"bitrate" : bitrate}) + '\n'
            clientsocket.sendall(payload.encode())

    # with open('./test_states/testPQ', 'w') as json_file:
    #     json.dump(data, json_file, indent=4)






if __name__ == "__main__":

    recv_commands()
    ss.close()
