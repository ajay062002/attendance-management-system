"""
main.py — entry point for the attendance management system.
Delegates to pipeline modules for dataset creation, training, and recognition.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pipeline.dataset import dataset_creation
from pipeline.trainer import train
from pipeline.recognizer import recognize


def main():
    print("Attendance Management System")
    print("1. Create Dataset")
    print("2. Train Classifier")
    print("3. Recognize (Webcam)")
    print("4. Recognize (Video)")
    print("5. Recognize (Images)")

    choice = input("\nSelect option: ").strip()

    if choice == '1':
        username = input("Enter person name: ").strip()
        params = ('', '', '', '', username, '')
        dataset_creation(params)

    elif choice == '2':
        params = ('', '', '', '', '', '', '', '')
        train(params)

    elif choice == '3':
        params = ('', '', '', '', '', '', '', '', '', '', '', '')
        recognize('w', params)

    elif choice == '4':
        vid_path = input("Enter video path: ").strip()
        params = ('', '', '', '', '', '', '', '', '', vid_path, '', '')
        recognize('v', params)

    elif choice == '5':
        img_path = input("Enter image folder path: ").strip()
        params = ('', '', '', '', '', '', '', img_path, '', '', '', '')
        recognize('i', params)

    else:
        print("Invalid option.")


if __name__ == '__main__':
    main()
