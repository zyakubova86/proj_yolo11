# Ultralytics YOLO 🚀, AGPL-3.0 license
import datetime

import cv2

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

from db import save_detection_to_db


class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
    specified region in a video stream. It supports both polygonal and linear regions for counting.

    Attributes:
        in_count (int): Counter for objects moving inward.
        out_count (int): Counter for objects moving outward.
        counted_ids (List[int]): List of IDs of objects that have been counted.
        classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class.
        region_initialized (bool): Flag indicating whether the counting region has been initialized.
        show_in (bool): Flag to control display of inward count.
        show_out (bool): Flag to control display of outward count.

    Methods:
        count_objects: Counts objects within a polygonal or linear region.
        store_classwise_counts: Initializes class-wise counts if not already present.
        display_counts: Displays object counts on the frame.
        count: Processes input data (frames or object tracks) and updates counts.

    Examples:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = counter.count(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
        self.region_initialized = False  # Bool variable for region initialization

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.resize_dst = self.CFG["resize_dst"]

        self.in_counted_ids = []
        self.out_counted_ids = []

    def count_objects(self, current_centroid, track_id, box, prev_position, cls, camera_id):
        """
        Counts objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (Tuple[float, float]): Current centroid values in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.
            camera_id (int): ID of the camera where the object is detected.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
        """

        # # TODO reset counters at the start of the day every day
        # reset counter parameters to 0 at 00:00:00

        if datetime.datetime.now().hour == 12 and datetime.datetime.now().minute == 0 and datetime.datetime.now().second == 0:
            self.in_count = 0
            self.out_count = 0
            self.classwise_counts = {}
        #     # self.model.predictor.trackers[0].reset_id()

        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:  # Linear region (defined as a line segment)
            line = self.LineString(self.region)  # Check if the line intersects the trajectory of the object
            if line.intersects(self.LineString([prev_position, current_centroid])):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        self.in_count += 1

                        self.classwise_counts[self.names[cls]]["IN"] += 1
                    else:  # Moving left
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                # Horizontal region: Compare y-coordinates to determine direction
                elif current_centroid[1] > prev_position[1]:  # Moving downward
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # Moving upward
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                # Determine motion direction for vertical or horizontal polygons
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                # TODO draw box and track trails when object is detected in polygonal region ===========

                if (
                        region_width < region_height
                        and current_centroid[0] > prev_position[0]
                        or region_width >= region_height
                        and current_centroid[1] > prev_position[1]
                ):  # Moving right
                    self.in_count += 1

                    # TODO save track_id to db ===============================================
                    self.in_counted_ids.append(track_id)

                    save_detection_to_db(track_id, self.names[cls], camera_id=camera_id)
                    # ========================================================================

                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # Moving left
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'IN' and 'OUT' counts to zero if the class is not already present.

        Examples:
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    @staticmethod
    def draw_text_multiline(img, text, pos, font, font_scale, font_thickness, text_color=(104, 31, 17),
                            bg_color=(255, 255, 255),
                            line_spacing=30, margin=15):
        x, y = pos
        lines = text.split('\n')

        # Calculate overall text block size
        line_heights = []
        text_width = 0
        for line in lines:
            (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            line_heights.append(line_height)
            text_width = max(text_width, line_width)

        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Draw background rectangle if bg_color is provided
        rect_top_left = (x - margin, y - margin)
        rect_bottom_right = (x + text_width + margin, y + total_height + margin)

        cv2.rectangle(img, rect_top_left, rect_bottom_right, bg_color, -1)

        # Draw each line of text
        for i, line in enumerate(lines):
            y_offset = sum(line_heights[:i]) + i * line_spacing
            cv2.putText(
                img, line, (x, y + y_offset + line_heights[i]),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA
            )

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        # labels_dict = {
        #     str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
        #     f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
        #     for key, value in self.classwise_counts.items()
        #     if value["IN"] != 0 or value["OUT"] != 0
        #
        # }
        #
        # if labels_dict:
        #     self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

        # txt = f"IN: {str(self.in_count)}\nOUT: {str(self.out_count)}\nTOTAL: {str(self.in_count + self.out_count)}"
        txt = f"IN {str(self.in_count)}\nOUT {str(self.out_count)}"
        self.draw_text_multiline(im0, txt, pos=(30, 60), font=0, font_scale=2, font_thickness=3)

    def count(self, im0, camera_name=None, camera_id=None):
        """
        Processes input data (frames or object tracks) and updates object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.
            camera_name (str): The name of the camera being processed, if applicable. Default is None.
            camera_id (int): The ID of the camera being processed, if applicable. Default is None.
        Returns:
            (numpy.ndarray): The processed image with annotations and count information.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=colors(int(19), True), thickness=self.line_width * 2
        )  # Draw region

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls, track_confidence in zip(self.boxes, self.track_ids, self.clss, self.track_confidences):
            track_confidence = int(track_confidence * 100)

            # Draw bounding box and counting region
            # self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.annotator.box_label(box, label=f"{track_id} | {track_confidence}", color=colors(10, True))
            self.store_tracking_history(track_id, box)  # Store track history
            self.store_classwise_counts(cls)  # store classwise counts in dict

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(10), True), track_thickness=self.line_width
            )
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            # store previous position of track for object counting

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]

            self.count_objects(current_centroid, track_id, box, prev_position, cls,
                               camera_id)  # Perform object counting

        self.display_counts(im0)  # Display the counts on the frame
        # self.display_output(im0)  # display output with base class function

        if self.CFG.get("show") and self.env_check:

            im0 = cv2.resize(im0, (self.resize_dst[0], self.resize_dst[1]))

            window_name = f"Tracking {camera_id}" if camera_name else "Tracking"

            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            cv2.imshow(window_name, im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()

                return

        return im0  # return output image for more usage
