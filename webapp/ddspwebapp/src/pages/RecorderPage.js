import React from "react";
// import "./App.css";
import MicRecorder from "mic-recorder-to-mp3";
import axios from "axios";

// COMPONENTS
import RecorderCmp from "./../components/RecorderCmp";

const Mp3Recorder = new MicRecorder({ bitRate: 128 });
const baseURL = "https://07da6b4b08b8.ngrok.io";

class RecorderPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isRecording: [false, false, false, false, false],
      blobURL_recorded: ["", "", "", "", ""],
      blob: ["", "", "", "", ""],
      blobURL_received: ["", "", "", "", ""],
      isBlocked: false,
    };
  }

  uploadAudio = async (audioBlob) => {
    let data = new FormData();
    data.append("file", audioBlob);
    return axios
      .post(`${baseURL}/audiorecog`, data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        return res;
      });
  };

  downloadAudio = async (num) => {
    return axios
      .get(`${baseURL}/audiorecog`, {
        responseType: "blob",
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        const blob = new Blob([res.data], { type: "audio/mp3" });
        const blobURL = window.URL.createObjectURL(blob);
        this.setState((prevState, props) => {
          const ـblobURL = prevState.blobURL_received.map((item, indx) => {
            if (indx === num) {
              return blobURL;
            } else {
              return item;
            }
          });
          return {
            blobURL_received: ـblobURL,
          };
        });
        return res;
      });
  };

  stertRecording = (num) => {
    if (this.state.isBlocked) {
      console.log("Permission Denied");
    } else {
      Mp3Recorder.start()
        .then(() => {
          this.setState((prevState, props) => {
            const isRecording = prevState.isRecording.map((item, indx) => {
              if (indx === num) {
                return true;
              } else {
                return item;
              }
            });
            return {
              isRecording,
            };
          });
        })
        .catch((e) => console.error(e));
    }
  };

  stopRecording = async (num) => {
    Mp3Recorder.stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const blobURL = URL.createObjectURL(blob);
        this.setState((prevState, props) => {
          const isRecording = prevState.isRecording.map((item, indx) => {
            if (indx === num) {
              return false;
            } else {
              return item;
            }
          });
          return {
            isRecording,
          };
        });
        this.setState((prevState, props) => {
          const ـblobURL = prevState.blobURL_recorded.map((item, indx) => {
            if (indx === num) {
              return blobURL;
            } else {
              return item;
            }
          });
          return {
            blobURL_recorded: ـblobURL,
          };
        });
        this.setState((prevState, props) => {
          const _blob = prevState.blob.map((item, indx) => {
            if (indx === num) {
              return blob;
            } else {
              return item;
            }
          });
          return {
            blob: _blob,
          };
        });
      })

      .catch((e) => console.log(e));
  };

  componentDidMount() {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(() => {
        console.log("Permission Granted");
        this.setState({ isBlocked: false });
      })
      .catch(() => {
        console.log("Permission Denied");
        this.setState({ isBlocked: true });
      });
  }

  render() {
    return (
      <div className="App">
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[0]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded[0]}
          num={0}
          uploadAudio={this.uploadAudio}
          downloadAudio={this.downloadAudio}
          blob={this.state.blob[0]}
          blobURL_received={this.state.blobURL_received[0]}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[1]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded[1]}
          num={1}
          uploadAudio={this.uploadAudio}
          downloadAudio={this.downloadAudio}
          blob={this.state.blob[1]}
          blobURL_received={this.state.blobURL_received[1]}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[2]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded[2]}
          num={2}
          uploadAudio={this.uploadAudio}
          downloadAudio={this.downloadAudio}
          blob={this.state.blob[2]}
          blobURL_received={this.state.blobURL_received[2]}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[3]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded[3]}
          num={3}
          uploadAudio={this.uploadAudio}
          downloadAudio={this.downloadAudio}
          blob={this.state.blob[3]}
          blobURL_received={this.state.blobURL_received[3]}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[4]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded[4]}
          num={4}
          uploadAudio={this.uploadAudio}
          downloadAudio={this.downloadAudio}
          blob={this.state.blob[4]}
          blobURL_received={this.state.blobURL_received[4]}
        />
      </div>
    );
  }
}

export default RecorderPage;
