import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import welch
from scipy.fft import rfft, rfftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed_sessions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "pca_svm"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 11)
]

RECORDINGS = [1, 2, 3]

FS = 256

EPOCHS_PER_WINDOW = 10

N_FEATURES = 224
N_PCA_COMPONENTS = 50

SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"


# ------------------------------------------------------------
# DEMO MODE
# ------------------------------------------------------------

# R1 = enrollment
# R2 = validation / threshold calibration
# R3 = unseen test

ENROLLMENT_RECORDING = 1
VALIDATION_RECORDING = 2
TEST_RECORDING = 3


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(eeg):
    """
    Input:
        (n_windows, 16, 256)

    Output:
        (n_windows, 224)

    16 channels × 14 features = 224 features

    Per channel:
        6 temporal features
        4 PSD features
        4 FFT features
    """

    features = []

    bands = [
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 30.0)
    ]

    for window in eeg:

        window_features = []

        for channel in window:

            # ------------------------------------------------
            # TEMPORAL FEATURES
            # ------------------------------------------------

            mean = np.mean(channel)
            std = np.std(channel)
            maximum = np.max(channel)
            minimum = np.min(channel)
            ptp = np.ptp(channel)
            variance = np.var(channel)

            window_features.extend([
                mean,
                std,
                maximum,
                minimum,
                ptp,
                variance
            ])

            # ------------------------------------------------
            # PSD FEATURES
            # ------------------------------------------------

            freqs, psd = welch(
                channel,
                fs=FS,
                nperseg=min(256, len(channel))
            )

            for low, high in bands:

                mask = (
                    (freqs >= low)
                    &
                    (freqs < high)
                )

                if np.any(mask):

                    value = np.mean(
                        psd[mask]
                    )

                else:

                    value = 0.0

                window_features.append(value)

            # ------------------------------------------------
            # FFT FEATURES
            # ------------------------------------------------

            fft_values = np.abs(
                rfft(channel)
            )

            fft_freqs = rfftfreq(
                len(channel),
                1 / FS
            )

            for low, high in bands:

                mask = (
                    (fft_freqs >= low)
                    &
                    (fft_freqs < high)
                )

                if np.any(mask):

                    value = np.mean(
                        fft_values[mask]
                    )

                else:

                    value = 0.0

                window_features.append(value)

        features.append(
            window_features
        )

    return np.asarray(features)


# ============================================================
# CREATE AUTHENTICATION WINDOWS
# ============================================================

def create_windows(epochs):
    """
    Combine 10 consecutive P300 epochs into
    one authentication window.

    Input:
        (1536, 16, 256)

    Output:
        (153, 16, 256)
    """

    n_complete = (
        epochs.shape[0]
        // EPOCHS_PER_WINDOW
    )

    usable = (
        n_complete
        * EPOCHS_PER_WINDOW
    )

    epochs = epochs[:usable]

    grouped = epochs.reshape(
        n_complete,
        EPOCHS_PER_WINDOW,
        epochs.shape[1],
        epochs.shape[2]
    )

    return grouped.mean(axis=1)


# ============================================================
# LOAD EEG RECORDING
# ============================================================

def load_recording(
    subject,
    recording
):

    path = (
        DATA_DIR
        / subject
        / f"recording_{recording}_epochs.npy"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"EEG file not found:\n{path}"
        )

    epochs = np.load(path)

    windows = create_windows(
        epochs
    )

    return windows


# ============================================================
# TRAIN SUBJECT AUTHENTICATOR
# ============================================================

def train_subject(
    claimed_subject
):

    print("\n")
    print("=" * 70)
    print("ENROLLMENT")
    print("=" * 70)

    print(
        f"Claimed identity: "
        f"{claimed_subject}"
    )

    print(
        f"Enrollment recording: "
        f"R{ENROLLMENT_RECORDING}"
    )

    X_all = []
    y_all = []

    # --------------------------------------------------------
    # Load all subjects
    # --------------------------------------------------------

    for subject in SUBJECTS:

        print(
            f"Loading enrollment: "
            f"{subject}"
        )

        windows = load_recording(
            subject,
            ENROLLMENT_RECORDING
        )

        features = extract_features(
            windows
        )

        label = (
            1
            if subject == claimed_subject
            else 0
        )

        labels = np.full(
            len(features),
            label
        )

        X_all.append(features)
        y_all.append(labels)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print(
        "\nFeature matrix:",
        X.shape
    )

    print(
        "Genuine samples:",
        np.sum(y == 1)
    )

    print(
        "Impostor samples:",
        np.sum(y == 0)
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    print(
        "Fitting StandardScaler..."
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_PCA_COMPONENTS,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    explained = (
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    print(
        f"PCA explained variance: "
        f"{explained * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print(
        "Training RBF SVM..."
    )

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA
    )

    svm.fit(
        X_pca,
        y
    )

    print(
        "✓ RBF SVM trained"
    )

    return (
        scaler,
        pca,
        svm,
        explained
    )


# ============================================================
# CALIBRATE THRESHOLD
# ============================================================

def calibrate_threshold(
    claimed_subject,
    scaler,
    pca,
    svm
):

    """
    Use validation recording R2 to determine
    the authentication threshold.

    Genuine:
        claimed subject, R2

    Impostor:
        all other subjects, R2
    """

    print("\n")
    print("=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)

    print(
        f"Validation recording: "
        f"R{VALIDATION_RECORDING}"
    )

    genuine_scores = []
    impostor_scores = []

    for subject in SUBJECTS:

        windows = load_recording(
            subject,
            VALIDATION_RECORDING
        )

        features = extract_features(
            windows
        )

        X_scaled = scaler.transform(
            features
        )

        X_pca = pca.transform(
            X_scaled
        )

        scores = svm.decision_function(
            X_pca
        )

        if subject == claimed_subject:

            genuine_scores.extend(
                scores
            )

        else:

            impostor_scores.extend(
                scores
            )

    genuine_scores = np.asarray(
        genuine_scores
    )

    impostor_scores = np.asarray(
        impostor_scores
    )

    # --------------------------------------------------------
    # Find threshold where FAR and FRR are closest
    # --------------------------------------------------------

    all_scores = np.concatenate([
        genuine_scores,
        impostor_scores
    ])

    thresholds = np.linspace(
        np.min(all_scores),
        np.max(all_scores),
        1000
    )

    best_threshold = thresholds[0]
    best_difference = float("inf")

    best_far = 1.0
    best_frr = 1.0

    for threshold in thresholds:

        far = np.mean(
            impostor_scores >= threshold
        )

        frr = np.mean(
            genuine_scores < threshold
        )

        difference = abs(
            far - frr
        )

        if difference < best_difference:

            best_difference = difference

            best_threshold = threshold

            best_far = far
            best_frr = frr

    print(
        f"Calibrated threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Validation FAR: "
        f"{best_far * 100:.2f}%"
    )

    print(
        f"Validation FRR: "
        f"{best_frr * 100:.2f}%"
    )

    return (
        best_threshold,
        genuine_scores,
        impostor_scores
    )


# ============================================================
# TEST AUTHENTICATION
# ============================================================

def authenticate(
    claimed_subject,
    test_subject,
    test_recording,
    scaler,
    pca,
    svm,
    threshold
):

    print("\n")
    print("=" * 70)
    print("AUTHENTICATION")
    print("=" * 70)

    print(
        f"Claimed identity: "
        f"{claimed_subject}"
    )

    print(
        f"Test EEG subject: "
        f"{test_subject}"
    )

    print(
        f"Unseen test recording: "
        f"R{test_recording}"
    )

    # IMPORTANT:
    # Test EEG is loaded from test_subject,
    # NOT from claimed_subject.
    #
    # This allows genuine and impostor
    # demonstrations.

    windows = load_recording(
        test_subject,
        test_recording
    )

    print(
        "Authentication windows:",
        len(windows)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print(
        "Extracting 224 EEG features..."
    )

    features = extract_features(
        windows
    )

    print(
        "Feature matrix:",
        features.shape
    )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        features
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_pca = pca.transform(
        X_scaled
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    scores = svm.decision_function(
        X_pca
    )

    # --------------------------------------------------------
    # Threshold decision
    # --------------------------------------------------------

    decisions = (
        scores >= threshold
    )

    accepted = int(
        np.sum(decisions)
    )

    rejected = int(
        len(decisions)
        - accepted
    )

    rate = (
        accepted
        / len(decisions)
        * 100
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        f"Accepted: {accepted}"
    )

    print(
        f"Rejected: {rejected}"
    )

    print(
        f"Authentication rate: "
        f"{rate:.2f}%"
    )

    return {
        "scores": scores,
        "decisions": decisions,
        "accepted": accepted,
        "rejected": rejected,
        "rate": rate
    }


# ============================================================
# GUI
# ============================================================

class AuthenticationDemo:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "P300 EEG Authentication System"
        )

        self.root.geometry(
            "1050x750"
        )

        self.root.configure(
            padx=20,
            pady=20
        )

        self.models = None
        self.threshold = None

        # ====================================================
        # HEADER
        # ====================================================

        header = tk.Frame(root)

        header.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            header,
            text="P300 EEG Authentication",
            font=("Arial", 27, "bold")
        ).pack()

        tk.Label(
            header,
            text=(
                "PCA + RBF SVM Cross-Session "
                "Biometric Authentication"
            ),
            font=("Arial", 13)
        ).pack(
            pady=5
        )

        # ====================================================
        # NOTEBOOK
        # ====================================================

        notebook = ttk.Notebook(root)

        notebook.pack(
            fill="both",
            expand=True
        )

        self.auth_tab = tk.Frame(
            notebook
        )

        self.performance_tab = tk.Frame(
            notebook
        )

        notebook.add(
            self.auth_tab,
            text="  Authentication  "
        )

        notebook.add(
            self.performance_tab,
            text="  Experimental Results  "
        )

        self.build_auth_tab()

        self.build_performance_tab()


    # ========================================================
    # AUTH TAB
    # ========================================================

    def build_auth_tab(self):

        tab = self.auth_tab

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        input_frame = tk.LabelFrame(
            tab,
            text="Authentication Input",
            font=("Arial", 13, "bold"),
            padx=20,
            pady=15
        )

        input_frame.pack(
            fill="x",
            pady=15
        )

        # ----------------------------------------------------
        # CLAIMED IDENTITY
        # ----------------------------------------------------

        tk.Label(
            input_frame,
            text="Claimed Identity:",
            font=("Arial", 12)
        ).grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="e"
        )

        self.user_combo = ttk.Combobox(
            input_frame,
            values=SUBJECTS,
            state="readonly",
            width=15
        )

        self.user_combo.set("A01")

        self.user_combo.grid(
            row=0,
            column=1,
            padx=10
        )

        # ----------------------------------------------------
        # TEST EEG SUBJECT
        # ----------------------------------------------------

        tk.Label(
            input_frame,
            text="Test EEG Subject:",
            font=("Arial", 12)
        ).grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="e"
        )

        self.test_subject_combo = ttk.Combobox(
            input_frame,
            values=SUBJECTS,
            state="readonly",
            width=15
        )

        self.test_subject_combo.set("A01")

        self.test_subject_combo.grid(
            row=1,
            column=1,
            padx=10
        )

        # ----------------------------------------------------
        # TEST RECORDING
        # ----------------------------------------------------

        tk.Label(
            input_frame,
            text="Unseen Test Session:",
            font=("Arial", 12)
        ).grid(
            row=2,
            column=0,
            padx=10,
            pady=10,
            sticky="e"
        )

        self.test_combo = ttk.Combobox(
            input_frame,
            values=[
                "Recording 3"
            ],
            state="readonly",
            width=15
        )

        self.test_combo.set(
            "Recording 3"
        )

        self.test_combo.grid(
            row=2,
            column=1,
            padx=10
        )

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        button_frame = tk.Frame(
            tab
        )

        button_frame.pack(
            pady=15
        )

        self.enroll_button = tk.Button(
            button_frame,
            text="1. ENROLL USER",
            width=22,
            height=2,
            font=("Arial", 13, "bold"),
            command=self.enroll
        )

        self.enroll_button.grid(
            row=0,
            column=0,
            padx=10
        )

        self.auth_button = tk.Button(
            button_frame,
            text="2. AUTHENTICATE",
            width=22,
            height=2,
            font=("Arial", 13, "bold"),
            command=self.authenticate,
            state="disabled"
        )

        self.auth_button.grid(
            row=0,
            column=1,
            padx=10
        )

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        pipeline = tk.LabelFrame(
            tab,
            text="Processing Pipeline",
            font=("Arial", 13, "bold"),
            padx=15,
            pady=15
        )

        pipeline.pack(
            fill="x",
            pady=10
        )

        pipeline_text = (
            "REAL EEG"
            "   →   "
            "10 P300 Epochs"
            "   →   "
            "224 Features"
            "   →   "
            "Scaling"
            "   →   "
            "PCA 224→50"
            "   →   "
            "RBF SVM"
            "   →   "
            "Decision"
        )

        tk.Label(
            pipeline,
            text=pipeline_text,
            font=("Arial", 12, "bold")
        ).pack()

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = tk.Label(
            tab,
            text="System ready.",
            font=("Arial", 13)
        )

        self.status.pack(
            pady=10
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        self.result_label = tk.Label(
            tab,
            text="",
            font=("Arial", 27, "bold")
        )

        self.result_label.pack(
            pady=10
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        self.metrics = tk.Label(
            tab,
            text=(
                "Enrollment: R1   |   "
                "Validation: R2   |   "
                "Test: R3"
            ),
            font=("Arial", 12)
        )

        self.metrics.pack(
            pady=10
        )

        self.progress = ttk.Progressbar(
            tab,
            orient="horizontal",
            length=700,
            mode="determinate"
        )

        self.progress.pack(
            pady=15
        )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        self.technical = tk.Label(
            tab,
            text=(
                "16 EEG channels  |  256 Hz  |  "
                "224 features  |  PCA: 50 components  |  "
                "RBF SVM"
            ),
            font=("Arial", 11)
        )

        self.technical.pack(
            pady=10
        )


    # ========================================================
    # ENROLL
    # ========================================================

    def enroll(self):

        try:

            subject = (
                self.user_combo.get()
            )

            # Set default genuine test subject
            self.test_subject_combo.set(
                subject
            )

            self.status.config(
                text=(
                    "Training subject "
                    "authenticator..."
                )
            )

            self.root.update()

            self.models = train_subject(
                subject
            )

            scaler, pca, svm, explained = (
                self.models
            )

            self.status.config(
                text=(
                    "Calibrating authentication "
                    "threshold..."
                )
            )

            self.root.update()

            (
                threshold,
                genuine,
                impostor
            ) = calibrate_threshold(
                subject,
                scaler,
                pca,
                svm
            )

            self.threshold = threshold

            self.auth_button.config(
                state="normal"
            )

            self.result_label.config(
                text="✓ USER ENROLLED"
            )

            self.status.config(
                text=(
                    "Enrollment and threshold "
                    "calibration complete."
                )
            )

            self.metrics.config(
                text=(
                    f"PCA variance retained: "
                    f"{explained * 100:.2f}%   |   "
                    f"Calibrated threshold: "
                    f"{threshold:.4f}"
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Enrollment Error",
                str(e)
            )


    # ========================================================
    # AUTHENTICATE
    # ========================================================

    def authenticate(self):

        try:

            if self.models is None:

                messagebox.showwarning(
                    "Enrollment Required",
                    "Enroll the user first."
                )

                return

            claimed_subject = (
                self.user_combo.get()
            )

            test_subject = (
                self.test_subject_combo.get()
            )

            test_recording = (
                TEST_RECORDING
            )

            self.status.config(
                text=(
                    "Processing unseen EEG "
                    "test session..."
                )
            )

            self.root.update()

            scaler, pca, svm, explained = (
                self.models
            )

            result = authenticate(
                claimed_subject,
                test_subject,
                test_recording,
                scaler,
                pca,
                svm,
                self.threshold
            )

            rate = result["rate"]

            accepted = result["accepted"]

            rejected = result["rejected"]

            # ------------------------------------------------
            # FINAL DECISION
            # ------------------------------------------------

            # Majority of authentication
            # windows must pass.

            if rate >= 50:

                self.result_label.config(
                    text="✓ AUTHENTICATED"
                )

                self.status.config(
                    text=(
                        f"The test EEG from "
                        f"{test_subject} matches "
                        f"the claimed identity "
                        f"{claimed_subject}."
                    )
                )

            else:

                self.result_label.config(
                    text="✗ REJECTED"
                )

                self.status.config(
                    text=(
                        f"The test EEG from "
                        f"{test_subject} does not "
                        f"match the claimed identity "
                        f"{claimed_subject}."
                    )
                )

            self.metrics.config(
                text=(
                    f"Claimed: {claimed_subject}   |   "
                    f"Test EEG: {test_subject}   |   "
                    f"Accepted: {accepted}/153   |   "
                    f"Rejected: {rejected}/153   |   "
                    f"Authentication rate: "
                    f"{rate:.2f}%   |   "
                    f"Threshold: {self.threshold:.4f}"
                )
            )

            self.progress["value"] = rate

        except Exception as e:

            messagebox.showerror(
                "Authentication Error",
                str(e)
            )


    # ========================================================
    # PERFORMANCE TAB
    # ========================================================

    def build_performance_tab(self):

        tab = self.performance_tab

        title = tk.Label(
            tab,
            text="Experimental Validation",
            font=("Arial", 20, "bold")
        )

        title.pack(
            pady=15
        )

        description = tk.Label(
            tab,
            text=(
                "Cross-session PCA + SVM results "
                "from the EEG authentication experiment"
            ),
            font=("Arial", 12)
        )

        description.pack(
            pady=5
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics_frame = tk.Frame(
            tab
        )

        metrics_frame.pack(
            pady=15
        )

        cards = [
            ("Mean ROC-AUC", "0.9388"),
            ("Mean EER", "13.15%"),
            ("Subjects", "10"),
            ("Sessions", "3")
        ]

        for i, (name, value) in enumerate(cards):

            frame = tk.LabelFrame(
                metrics_frame,
                text=name,
                padx=20,
                pady=10
            )

            frame.grid(
                row=0,
                column=i,
                padx=8
            )

            tk.Label(
                frame,
                text=value,
                font=("Arial", 18, "bold")
            ).pack()

        # ----------------------------------------------------
        # GRAPH BUTTONS
        # ----------------------------------------------------

        graph_frame = tk.Frame(
            tab
        )

        graph_frame.pack(
            pady=20
        )

        graphs = [
            (
                "ROC Curve",
                "roc_curve.png"
            ),
            (
                "FAR / FRR",
                "far_frr_curve.png"
            ),
            (
                "EER by Session",
                "eer_by_recording_pair.png"
            ),
            (
                "Score Distribution",
                "score_distribution.png"
            )
        ]

        for i, (name, filename) in enumerate(
            graphs
        ):

            button = tk.Button(
                graph_frame,
                text=name,
                width=20,
                height=2,
                font=("Arial", 11, "bold"),
                command=lambda f=filename:
                    self.show_graph(f)
            )

            button.grid(
                row=0,
                column=i,
                padx=5
            )

        self.graph_label = tk.Label(
            tab,
            text=(
                "Select a graph above to "
                "display the experimental result."
            ),
            font=("Arial", 12)
        )

        self.graph_label.pack(
            pady=15
        )


    # ========================================================
    # SHOW GRAPH
    # ========================================================

    def show_graph(
        self,
        filename
    ):

        path = (
            RESULTS_DIR
            / filename
        )

        if not path.exists():

            messagebox.showerror(
                "Graph Missing",
                f"Could not find:\n{path}"
            )

            return

        try:

            image = plt.imread(
                str(path)
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.imshow(
                image
            )

            ax.axis(
                "off"
            )

            self.graph_label.config(
                text=filename
            )

            # Remove previous canvas
            for widget in self.performance_tab.winfo_children():

                if isinstance(
                    widget,
                    FigureCanvasTkAgg
                ):

                    widget.get_tk_widget().destroy()

            canvas = FigureCanvasTkAgg(
                fig,
                master=self.performance_tab
            )

            canvas.draw()

            canvas.get_tk_widget().pack(
                fill="both",
                expand=True,
                pady=10
            )

        except Exception as e:

            messagebox.showerror(
                "Graph Error",
                str(e)
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("P300 EEG AUTHENTICATION DEMO")
    print("=" * 70)

    print(
        "\nDemo protocol:"
    )

    print(
        "Enrollment : Recording 1"
    )

    print(
        "Validation : Recording 2"
    )

    print(
        "Test       : Recording 3"
    )

    print(
        "\nStarting GUI..."
    )

    root = tk.Tk()

    app = AuthenticationDemo(
        root
    )

    root.mainloop()
