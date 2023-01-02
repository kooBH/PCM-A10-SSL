# PCM-A10-SSL
![image](https://user-images.githubusercontent.com/39723411/208828983-9c9c6283-dbed-4b4a-9334-c5d6b851a00b.png)


 For autonomous rescue drones, the direction of audio arrival estimation is useful in disaster situations. In such situations, audio information can be acquired regardless of visual obstacles. And to be run on mobile drones, a deep learning based model must be light-weighted. This code provides a model which is trained to detect the direction of a target source. This code is written in Python, based on Pytorch.

---

Sound Source Localization for [PCM-A10](https://www.sony.co.kr/electronics/voice-recorders/pcm-a10) Microphone.

# Usage

```
python run.py -i <directory of input files > --device <cuda:0 or cuda:1 or cpu>
```


# Acknowledgement
이 연구는 2022년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구의 결과물임 (No.1711152445, 인명 구조용 드론을 위한 영상/음성 인지 기술 고도화)

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.1711152445, Advanced Audio-Visual Perception for Autonomous Rescue Drones)
