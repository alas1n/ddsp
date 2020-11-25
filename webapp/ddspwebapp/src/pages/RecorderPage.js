import React from "react";
import MicRecorder from "mic-recorder-to-mp3";
import axios from "axios";

// COMPONENTS
import RecorderCmp from "./../components/RecorderCmp";

const Mp3Recorder = new MicRecorder({ bitRate: 128 });
const baseURL = "http://127.0.0.1:5000";

class RecorderPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isRecording: false,
      blobURL_recorded: "",
      blob: "",
      blobURL_received: ["", "", "", "", ""],
      isBlocked: false,
    };
  }

  startRecording = () => {
    if (this.state.isBlocked) {
      console.log("Permission Denied");
    } else {
      Mp3Recorder.start()
        .then(() => {
          this.setState((prevState, props) => {
            return {
              ...prevState,
              isRecording: true,
            };
          });
        })
        .catch((e) => console.error(e));
    }
  };

  stopRecording = async () => {
    Mp3Recorder.stop()
      .getMp3()
      .then(([buffer, blob]) => {
        const blobURL = URL.createObjectURL(blob);
        this.setState((prevState, props) => {
          return {
            ...prevState,
            isRecording: false,
            blobURL_recorded: blobURL,
            blob,
          };
        });
      })
      .catch((e) => console.log(e));
  };

  uploadAudio = async (audioBlob) => {
    let data = new FormData();
    data.append("file", audioBlob);
    return axios
      .post(`${baseURL}/audiorecorder/`, data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        return res;
      });
  };

  downloadAudio = async (num, model) => {
    await axios
      .get(`${baseURL}/timbretransfer/`, {
        params: { model, num },
        responseType: "blob",
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        const blob = new Blob([res.data], { type: "audio/mp3" });
        const blobURL = URL.createObjectURL(blob);
        console.log("blobURL", blobURL);
        this.setState((prevState, props) => {
          const ـblobURL = prevState.blobURL_received.map((item, indx) => {
            if (indx === num) {
              return blobURL;
            } else {
              return item;
            }
          });
          return {
            ...prevState,
            blobURL_received: ـblobURL,
          };
        });
        // return res;
      });
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
          startRecording={this.startRecording}
          isRecording={this.state.isRecording}
          stopRecording={this.stopRecording}
          blobURL={this.state.blobURL_recorded}
          num={0}
          uploadAudio={this.uploadAudio}
          blob={this.state.blob}
          // downloadAudio={this.downloadAudio}
          // blobURL_received={this.state.blobURL_received}
        />
        <button
          onClick={async () => {
            await this.downloadAudio(0, "Violin");
            console.log("Audio downloaded");
          }}
        >
          Violin
        </button>
        {this.state.blobURL_received[0] ? (
          <audio src={this.state.blobURL_received[0]} controls="controls" />
        ) : undefined}
        <button
          onClick={async () => {
            await this.downloadAudio(1, "Flute");
            console.log("Audio downloaded");
          }}
        >
          Flute
        </button>
        <audio src={this.state.blobURL_received[1]} controls="controls" />
        <button
          onClick={async () => {
            await this.downloadAudio(2, "Flute2");
            console.log("Audio downloaded");
          }}
        >
          Flute2
        </button>
        <audio src={this.state.blobURL_received[2]} controls="controls" />
        <button
          onClick={async () => {
            await this.downloadAudio(3, "Trumpet");
            console.log("Audio downloaded");
          }}
        >
          Trumpet
        </button>
        <audio src={this.state.blobURL_received[3]} controls="controls" />
        <button
          onClick={async () => {
            await this.downloadAudio(4, "Tenor_Saxophone");
            console.log("Audio downloaded");
          }}
        >
          Tenor_Saxophone
        </button>
        <audio src={this.state.blobURL_received[4]} controls="controls" />
        <button
          onClick={() => {
            console.log(this.state);
            this.setState((prevState) => {
              return {
                ...prevState,
                blobURL_received: ["", "", "", "", ""],
              };
            });
            console.log(this.state);
          }}
        >
          state
        </button>
      </div>
    );
  }
}

export default RecorderPage;
