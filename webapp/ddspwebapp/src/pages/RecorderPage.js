import React from "react";
// import "./App.css";
import MicRecorder from "mic-recorder-to-mp3";

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
        const blobURLR = URL.createObjectURL(blob);
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
          const blobURL = prevState.blobURL.map((item, indx) => {
            if (indx === num) {
              return blobURLR;
            } else {
              return item;
            }
          });
          return {
            blobURL,
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
        {/* <header className="App-header">
          <button
            onClick={() => this.stertRecording(0)}
            disabled={this.state.isRecording[0]}
          >
            Record
          </button>
          <button
            onClick={() => this.stopRecording(0)}
            disabled={!this.state.isRecording[0]}
          >
            Stop
          </button>
          <audio src={this.state.blobURL[0]} controls="controls" />
        </header> */}
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
      </div>
    );
  }
}

export default RecorderPage;
