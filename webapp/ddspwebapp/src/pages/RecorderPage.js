import React from "react";
// import "./App.css";
import MicRecorder from "mic-recorder-to-mp3";
import axios from "axios";

// COMPONENTS
import RecorderCmp from "./../components/RecorderCmp";

const Mp3Recorder = new MicRecorder({ bitRate: 128 });

class RecorderPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isRecording: [false, false, false, false, false],
      blobURL: ["", "", "", "", ""],
      isBlocked: false,
    };
  }

  // postToServer = async () => {
  //   await axios
  //     .post(`https://9c96ceeee5c7.ngrok.io/items/${this.state.blobObj}`)
  //     .then((res) => {
  //       const persons = res.data;
  //       // this.setState({ persons });
  //       console.log(persons);
  //     });
  // };
  uploadAudio = async (audioBlob) => {
    console.log(audioBlob);
    let data = new FormData();
    console.log(data);
    data.append("file", audioBlob);
    console.log(data);
    return axios
      .post(`https://9c96ceeee5c7.ngrok.io/audiorecog`, data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res);
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

  stopRecording = (num) => {
    Mp3Recorder.stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const blobURL = URL.createObjectURL(blob);
        const blobObj = blob;
        console.log(blob);
        this.setState((prevState) => {
          return {
            ...prevState,
            blobObj,
          };
        });
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
          const ـblobURL = prevState.blobURL.map((item, indx) => {
            if (indx === num) {
              return blobURL;
            } else {
              return item;
            }
          });
          return {
            blobURL: ـblobURL,
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
          blobURL={this.state.blobURL[0]}
          num={0}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[1]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL[1]}
          num={1}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[2]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL[2]}
          num={2}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[3]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL[3]}
          num={3}
        />
        <RecorderCmp
          stertRecording={this.stertRecording}
          isRecording={this.state.isRecording[4]}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL[4]}
          num={4}
        />
        <button
          onClick={() => {
            this.uploadAudio(this.state.blobObj);
            console.log("send");
          }}
        >
          send
        </button>
        <button
          onClick={() => {
            console.log("record 1:", this.state.blobURL[0]);
            console.log("blob_Obj:", this.state.blobObj);
          }}
        >
          print
        </button>
      </div>
    );
  }
}

export default RecorderPage;
