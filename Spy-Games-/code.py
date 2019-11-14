# --------------
##File path for the file 
# print(file_path)
def read_file(path):
    sentence = '' 
    with open(file_path , 'r') as f:
        sentence = f.read()
    return sentence
sample_message = read_file(file_path)
#Code starts here


# --------------
#Code starts here
message_1 = read_file(file_path_1)
print(message_1)
message_2 = read_file(file_path_2)
print(message_2)

def fuse_msg(message_a,message_b):
    quotient = int(message_b)//int(message_a)
    return str(quotient)
secret_msg_1 = fuse_msg(message_1 , message_2)




# --------------
#Code starts here
message_3 = read_file(file_path_3)
message_3
def substitute_msg(message_c):

           if (message_c == 'Red'):
               sub =('Army Genera')
           elif (message_c == 'Green'):
               sub = ('Data Scientist')
           elif (message_c == 'Blue'):
               sub = ('Marine Biologist')
           return sub

secret_msg_2 = substitute_msg(message_3)


# --------------
# File path for message 4  and message 5
file_path_4
file_path_5

#Code starts here
message_4 = read_file(file_path_4)
print(message_4)
message_5 = read_file(file_path_5)
print(message_5)

def compare_msg(message_d ,message_e):
    a_list = (message_d.split())
    b_list = (message_e.split())
    c_list = [x for x in a_list if x not in b_list]
    final_msg = ' '.join(c_list)
    return final_msg
secret_msg_3 = compare_msg(message_4 , message_5)
secret_msg_3




# --------------
#Code starts here
import re
message_6 = read_file(file_path_6)
print(message_6)
def extract_msg(message_f):
    a_list = re.findall(r'\w+', message_f)
    even_word = lambda x:(len(x) % 2 == 0)
    b_list = list(filter(even_word,a_list))
    final_msg = ' '.join(b_list)
    return final_msg
secret_msg_4 = extract_msg(message_6)
secret_msg_4

    


# --------------
#Secret message parts in the correct order
message_parts=[secret_msg_3, secret_msg_1, secret_msg_4, secret_msg_2]


final_path= user_data_dir + '/secret_message.txt'

#Code starts here
secret_msg = ' '.join(message_parts)
print(secret_msg)
def write_file(secret_msg,path):
    f =open(path,'a+')
    f.write(secret_msg)
    f.close()
write_file(secret_msg,final_path)
print(secret_msg)


