import detect

# Ран нужно переписать так что бы он возврашал значение можно передать опции source
# from yolov5 import suchev_detect

video_path = r'video/AM-STO-vorota_20220411-111346--20220411-111351.avi'  # 384x640
rtsp_kia = "http://10.30.19.2:555/ohpyDwj2?container=flv&stream=main"  # 384x640
rtsp_fgr = 'rtsp://steadycontrol:0ahFFrfs@10.30.1.15:554/live/main'  # 384x640
detect.run(source=rtsp_kia)
