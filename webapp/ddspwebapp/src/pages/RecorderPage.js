import React from "react";
// import "./App.css";
import MicRecorder from "mic-recorder-to-mp3";

const Mp3Recorder = new MicRecorder({ bitRate: 128 });

class RecorderPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isRecording: false,
      blobURL: "",
      isBlocked: false,
    };
  }

  stertRecording = () => {
    if (this.state.isBlocked) {
      console.log("Permission Denied");
    } else {
      Mp3Recorder.start()
        .then(() => {
          this.setState({ isRecording: true });
        })
        .catch((e) => console.error(e));
    }
  };

  stopRecording = () => {
    Mp3Recorder.stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const blobURL = URL.createObjectURL(blob);
        this.setState({ blobURL, isRecording: false });
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
        <header className="App-header">
          <button
            onClick={this.stertRecording}
            disabled={this.state.isRecording}
          >
            Record
          </button>
          <button
            onClick={this.stopRecording}
            disabled={!this.state.isRecording}
          >
            Stop
          </button>
          <audio src={this.state.blobURL} controls="controls" />
        </header>
      </div>
    );
  }
}

export default RecorderPage;
