import time

import cv2

from ultralytics.solutions import ObjectCounter


def process_and_count(source, camera_name, camera_id, model_name, region, line_width=3, resize_dst=(800, 450),
                      frame_skip=5, max_retries=10):
    retry_count = 0

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    counter = ObjectCounter(model=model_name, region_initialized=True, region=region, show=True,
                            line_width=line_width, resize_dst=resize_dst)

    frame_count = 0

    try:
        while retry_count < max_retries:

            # retry implementation
            if not cap.isOpened():
                print(f"{camera_name} connection failed. Retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(2)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            success, frame = cap.read()

            if not success:
                print(f"{camera_name} lost connection. Reconnecting...({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(2)

                cap.release()
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # internal buffer will now store only 0 frames NOT SUITABLE TO RTSP STREAM

                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                continue

            retry_count = 0  # reset retry count after successful connection

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            counter.count(frame, camera_name=camera_name, camera_id=camera_id)

    except KeyboardInterrupt:
        print("Existing via keyboard")

    except Exception as e:
        print(f"Error in {camera_name}: {e}")

    finally:

        cap.release()
        cv2.destroyAllWindows()


def run():
    from multiprocessing import Process
    from db import get_cameras_data

    model_name = "runs/detect/ds8_1_3_l3/weights/best.pt"
    roi_points = [(1000, 250), (1000, 900), (1300, 900), (1300, 250)]

    cameras = get_cameras_data()
    if cameras is None:
        print("No cameras found in the database.")
        return

    processes = []
    for camera in cameras:
        # print(camera)
        camera_id, rtsp_link, cam_name = camera[:3]

        detection_process = Process(target=process_and_count,
                                    args=(rtsp_link, cam_name, camera_id, model_name, roi_points))
        processes.append(detection_process)
        detection_process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    run()
