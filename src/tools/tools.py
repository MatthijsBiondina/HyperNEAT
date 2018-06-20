import time

def log( object_instance, message ):
    string = ( str(object_instance) + '  ' + time.ctime(time.time()) + '  \n\t' +
               str(message).replace('\n','\n\t\t')
               )
    if True: # do you want to write log?
        with open('../log/log.txt','a') as f:
            f.write(string + '\n\n')
    if True: # print?
        print(string)
