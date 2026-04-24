#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QPixmap>
#include <QFrame>
#include <QGroupBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QSettings>
#include <QFileInfo>
#include <QFile>
#include <QHttpMultiPart>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QCoreApplication>
#include <QDir>
#include <QTimer>
#include <QUrl>
#include <QStatusBar>
#include <QFont>
#include <QColor>
#include <functional>

// ---------------------------------------------------------------------------
// AMD_GUI – main application window
// ---------------------------------------------------------------------------
class AMD_GUI : public QMainWindow {
public:
    // ── Sidebar ─────────────────────────────────────────────────────────────
    QLineEdit   *nameInput;
    QSpinBox    *ageInput;
    QPushButton *uploadBtn;
    QPushButton *themeBtn;
    QLabel      *backendStatusLabel;
    QLabel      *scanCountLabel;

    // ── Analysis tab ────────────────────────────────────────────────────────
    QTabWidget   *tabWidget;
    QLabel       *fundusLabel;
    QLabel       *camsLabel;
    QLabel       *predictionBadge;
    QProgressBar *confidenceBar;
    QLabel       *confidenceValueLabel;
    QLabel       *riskLabel;
    QLabel       *modelInfoLabel;
    QLabel       *accuracyLabel;
    QLabel       *precisionLabel;
    QLabel       *recallLabel;
    QLabel       *f1Label;

    // ── Records tab ─────────────────────────────────────────────────────────
    QTableWidget *patientTable;
    QLineEdit    *searchInput;
    QLabel       *recordsStatsLabel;

    // ── State ────────────────────────────────────────────────────────────────
    bool                  isDarkMode;
    QSettings            *settings;
    QNetworkAccessManager *networkManager;
    QProcess             *backendProcess;
    bool                  backendStartedByGui;
    QList<QJsonObject>    allPatients;   // local cache for search filtering

    // ────────────────────────────────────────────────────────────────────────
    AMD_GUI() {
        setWindowTitle("AMD Detection System — Retinal Analysis Platform");
        resize(1280, 800);

        settings            = new QSettings("AMD_Detection", "AMD_GUI", this);
        isDarkMode          = settings->value("darkMode", false).toBool();
        networkManager      = new QNetworkAccessManager(this);
        backendProcess      = new QProcess(this);
        backendStartedByGui = false;

        setupUI();

        if (isDarkMode) applyDarkMode();
        else            applyLightMode();

        // Periodic backend health check
        auto *statusTimer = new QTimer(this);
        connect(statusTimer, &QTimer::timeout, this, [this]() { refreshBackendStatus(); });
        statusTimer->start(5000);
        refreshBackendStatus();
    }

    ~AMD_GUI() override {
        if (backendStartedByGui && backendProcess->state() != QProcess::NotRunning) {
            backendProcess->terminate();
            if (!backendProcess->waitForFinished(1500))
                backendProcess->kill();
        }
    }

private:
    // ── UI construction ─────────────────────────────────────────────────────
    void setupUI() {
        QWidget *central = new QWidget(this);
        setCentralWidget(central);

        QHBoxLayout *root = new QHBoxLayout(central);
        root->setSpacing(0);
        root->setContentsMargins(0, 0, 0, 0);

        // ── LEFT SIDEBAR ──────────────────────────────────────────────────
        QWidget *sidebar = new QWidget();
        sidebar->setObjectName("sidebar");
        sidebar->setFixedWidth(275);

        QVBoxLayout *sideLayout = new QVBoxLayout(sidebar);
        sideLayout->setContentsMargins(16, 20, 16, 20);
        sideLayout->setSpacing(10);

        // Logo / title
        QLabel *appTitle = new QLabel("🔬  AMD Detect");
        appTitle->setObjectName("appTitle");
        sideLayout->addWidget(appTitle);

        QLabel *appSub = new QLabel("Retinal Analysis Platform");
        appSub->setObjectName("appSubtitle");
        sideLayout->addWidget(appSub);

        auto *sep1 = new QFrame(); sep1->setFrameShape(QFrame::HLine);
        sep1->setObjectName("separator");
        sideLayout->addWidget(sep1);

        // Patient info form
        QGroupBox *patientGroup = new QGroupBox("Patient Information");
        patientGroup->setObjectName("patientGroup");
        QFormLayout *form = new QFormLayout(patientGroup);
        form->setSpacing(8);

        nameInput = new QLineEdit();
        nameInput->setPlaceholderText("Patient name…");
        nameInput->setObjectName("sidebarInput");
        form->addRow("Name:", nameInput);

        ageInput = new QSpinBox();
        ageInput->setRange(0, 120);
        ageInput->setValue(50);
        ageInput->setObjectName("sidebarInput");
        form->addRow("Age:", ageInput);

        sideLayout->addWidget(patientGroup);

        // Primary upload/analyse button
        uploadBtn = new QPushButton("📁   Upload & Analyse");
        uploadBtn->setObjectName("primaryBtn");
        uploadBtn->setMinimumHeight(44);
        sideLayout->addWidget(uploadBtn);

        auto *sep2 = new QFrame(); sep2->setFrameShape(QFrame::HLine);
        sep2->setObjectName("separator");
        sideLayout->addWidget(sep2);

        // Status section
        QLabel *statusHdr = new QLabel("SYSTEM STATUS");
        statusHdr->setObjectName("sectionHeader");
        sideLayout->addWidget(statusHdr);

        backendStatusLabel = new QLabel("● Checking…");
        backendStatusLabel->setObjectName("statusLabel");
        backendStatusLabel->setWordWrap(true);
        sideLayout->addWidget(backendStatusLabel);

        scanCountLabel = new QLabel("Total scans: —");
        scanCountLabel->setObjectName("infoLabel");
        sideLayout->addWidget(scanCountLabel);

        sideLayout->addStretch();

        // Theme toggle
        themeBtn = new QPushButton(isDarkMode ? "☀   Light Mode" : "🌙   Dark Mode");
        themeBtn->setObjectName("secondaryBtn");
        sideLayout->addWidget(themeBtn);

        root->addWidget(sidebar);

        // Thin vertical rule
        auto *vRule = new QFrame(); vRule->setFrameShape(QFrame::VLine);
        vRule->setObjectName("vSeparator");
        root->addWidget(vRule);

        // ── RIGHT PANEL: TABS ─────────────────────────────────────────────
        tabWidget = new QTabWidget();
        tabWidget->setObjectName("mainTabs");

        // ── TAB 1: Analysis ───────────────────────────────────────────────
        QWidget     *analysisTab    = new QWidget();
        QVBoxLayout *analysisLayout = new QVBoxLayout(analysisTab);
        analysisLayout->setContentsMargins(16, 16, 16, 16);
        analysisLayout->setSpacing(12);

        // Image display row
        QHBoxLayout *imgRow = new QHBoxLayout();
        imgRow->setSpacing(16);

        auto makeImgGroup = [](const QString &title, QLabel *&lbl) {
            QGroupBox   *grp = new QGroupBox(title);
            grp->setObjectName("imageGroup");
            QVBoxLayout *vb  = new QVBoxLayout(grp);
            lbl = new QLabel("No image loaded");
            lbl->setAlignment(Qt::AlignCenter);
            lbl->setFixedSize(330, 330);
            lbl->setScaledContents(true);
            lbl->setObjectName("imageDisplay");
            vb->addWidget(lbl);
            return grp;
        };

        imgRow->addWidget(makeImgGroup("Fundus Image", fundusLabel));
        imgRow->addWidget(makeImgGroup("Saliency Map (CAM)", camsLabel));
        imgRow->addStretch();
        analysisLayout->addLayout(imgRow);

        // Results group
        QGroupBox   *resultsGrp    = new QGroupBox("Analysis Results");
        resultsGrp->setObjectName("resultsGroup");
        QVBoxLayout *resultsLayout = new QVBoxLayout(resultsGrp);
        resultsLayout->setSpacing(10);

        // Prediction badge + risk level
        QHBoxLayout *badgeRow = new QHBoxLayout();
        predictionBadge = new QLabel("Awaiting Analysis");
        predictionBadge->setAlignment(Qt::AlignCenter);
        predictionBadge->setObjectName("predictionBadge");
        predictionBadge->setMinimumSize(200, 44);
        badgeRow->addWidget(predictionBadge);

        riskLabel = new QLabel("Risk: —");
        riskLabel->setAlignment(Qt::AlignCenter);
        riskLabel->setObjectName("riskLabel");
        riskLabel->setMinimumSize(160, 44);
        badgeRow->addWidget(riskLabel);
        badgeRow->addStretch();
        resultsLayout->addLayout(badgeRow);

        // Confidence bar
        QHBoxLayout *confRow = new QHBoxLayout();
        QLabel *confLbl = new QLabel("Confidence:");
        confLbl->setMinimumWidth(95);
        confRow->addWidget(confLbl);

        confidenceBar = new QProgressBar();
        confidenceBar->setRange(0, 100);
        confidenceBar->setValue(0);
        confidenceBar->setObjectName("confidenceBar");
        confidenceBar->setMinimumHeight(24);
        confidenceBar->setTextVisible(false);
        confRow->addWidget(confidenceBar);

        confidenceValueLabel = new QLabel("—");
        confidenceValueLabel->setMinimumWidth(65);
        confidenceValueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        confidenceValueLabel->setObjectName("metricValue");
        confRow->addWidget(confidenceValueLabel);
        resultsLayout->addLayout(confRow);

        // Metrics grid
        QGridLayout *metricsGrid = new QGridLayout();
        metricsGrid->setHorizontalSpacing(24);
        metricsGrid->setVerticalSpacing(6);

        auto addMetricPair = [&](int row, int col, const QString &title, QLabel *&valueLabel) {
            QLabel *hdr = new QLabel(title);
            hdr->setObjectName("metricHeader");
            metricsGrid->addWidget(hdr, row, col * 2);
            valueLabel = new QLabel("—");
            valueLabel->setObjectName("metricValue");
            metricsGrid->addWidget(valueLabel, row, col * 2 + 1);
        };

        addMetricPair(0, 0, "Model:",     modelInfoLabel);
        addMetricPair(0, 1, "Accuracy:",  accuracyLabel);
        addMetricPair(1, 0, "Precision:", precisionLabel);
        addMetricPair(1, 1, "Recall:",    recallLabel);
        addMetricPair(2, 0, "F1 Score:",  f1Label);
        resultsLayout->addLayout(metricsGrid);

        analysisLayout->addWidget(resultsGrp);
        tabWidget->addTab(analysisTab, "🔍   Analysis");

        // ── TAB 2: Patient Records ─────────────────────────────────────────
        QWidget     *recordsTab    = new QWidget();
        QVBoxLayout *recordsLayout = new QVBoxLayout(recordsTab);
        recordsLayout->setContentsMargins(16, 16, 16, 16);
        recordsLayout->setSpacing(10);

        // Control bar: search + refresh
        QHBoxLayout *ctrlRow = new QHBoxLayout();
        QLabel *searchIcon = new QLabel("🔎");
        ctrlRow->addWidget(searchIcon);

        searchInput = new QLineEdit();
        searchInput->setPlaceholderText("Search by patient name…");
        searchInput->setObjectName("searchBar");
        ctrlRow->addWidget(searchInput);

        QPushButton *refreshBtn = new QPushButton("⟳   Refresh");
        refreshBtn->setObjectName("secondaryBtn");
        refreshBtn->setMaximumWidth(120);
        ctrlRow->addWidget(refreshBtn);
        recordsLayout->addLayout(ctrlRow);

        // Stats bar
        recordsStatsLabel = new QLabel("No records loaded.");
        recordsStatsLabel->setObjectName("statsLabel");
        recordsLayout->addWidget(recordsStatsLabel);

        // Patient table
        patientTable = new QTableWidget();
        patientTable->setObjectName("patientTable");
        patientTable->setColumnCount(6);
        patientTable->setHorizontalHeaderLabels({"ID", "Patient Name", "Age", "Diagnosis", "Confidence", "Date"});
        patientTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        patientTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        patientTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
        patientTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        patientTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        patientTable->setSortingEnabled(true);
        patientTable->verticalHeader()->setVisible(false);
        patientTable->setAlternatingRowColors(true);
        patientTable->setShowGrid(false);
        recordsLayout->addWidget(patientTable);

        tabWidget->addTab(recordsTab, "📋   Patient Records");

        // ── TAB 3: About ──────────────────────────────────────────────────
        QWidget     *aboutTab    = new QWidget();
        QVBoxLayout *aboutLayout = new QVBoxLayout(aboutTab);
        aboutLayout->setContentsMargins(24, 24, 24, 24);
        aboutLayout->setSpacing(10);

        QLabel *aboutTitle = new QLabel("AMD Detection System");
        aboutTitle->setObjectName("aboutTitle");
        aboutLayout->addWidget(aboutTitle);

        QLabel *aboutText = new QLabel(
            "<p><b>Age-Related Macular Degeneration (AMD) Detection</b></p>"
            "<p>This system uses a <b>Vision Transformer (ViT-B16)</b> deep learning model to classify "
            "retinal fundus images as <b>Normal</b> or <b>AMD</b>.</p><hr/>"
            "<p><b>Features</b></p>"
            "<ul>"
            "<li>AI-powered retinal image classification</li>"
            "<li>Gradient-based saliency map (CAM) visualisation</li>"
            "<li>Persistent patient record database (SQLite)</li>"
            "<li>Confidence scoring &amp; three-tier risk assessment</li>"
            "<li>Model performance metrics: Accuracy, Precision, Recall, F1</li>"
            "<li>Searchable &amp; sortable patient history</li>"
            "</ul><hr/>"
            "<p><b>How to use</b></p>"
            "<ol>"
            "<li>Enter the patient name and age in the left sidebar</li>"
            "<li>Click <i>Upload &amp; Analyse</i> and select a fundus image</li>"
            "<li>Review the diagnosis, confidence bar, and saliency map</li>"
            "<li>Every scan is auto-saved — view history in <i>Patient Records</i></li>"
            "<li>Click any row in the records table to reload that fundus image</li>"
            "</ol>"
        );
        aboutText->setWordWrap(true);
        aboutText->setObjectName("aboutText");
        aboutLayout->addWidget(aboutText);
        aboutLayout->addStretch();

        tabWidget->addTab(aboutTab, "ℹ   About");

        root->addWidget(tabWidget, 1);

        // Status bar
        statusBar()->showMessage("Ready — upload a fundus image to begin analysis.");

        // ── CONNECTIONS ──────────────────────────────────────────────────
        connect(uploadBtn,  &QPushButton::clicked,       this, &AMD_GUI::uploadImage);
        connect(themeBtn,   &QPushButton::clicked,       this, &AMD_GUI::toggleTheme);
        connect(refreshBtn, &QPushButton::clicked,       this, &AMD_GUI::refreshPatientRecords);
        connect(searchInput, &QLineEdit::textChanged,    this, &AMD_GUI::filterRecords);
        connect(patientTable, &QTableWidget::cellClicked, this, &AMD_GUI::onRecordSelected);
    }

    // ── Slots / helpers ──────────────────────────────────────────────────────

    void uploadImage() {
        const QString fileName = QFileDialog::getOpenFileName(
            this, "Select Fundus Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        );
        if (fileName.isEmpty()) return;

        fundusLabel->setPixmap(QPixmap(fileName));
        camsLabel->setText("Processing…");
        predictionBadge->setText("Analysing…");
        predictionBadge->setStyleSheet("");
        riskLabel->setText("Risk: —");
        riskLabel->setStyleSheet("");
        confidenceBar->setValue(0);
        confidenceValueLabel->setText("—");
        statusBar()->showMessage("Analysing: " + QFileInfo(fileName).fileName());
        ensureBackendAndPredict(fileName);
    }

    void ensureBackendAndPredict(const QString &fileName) {
        QNetworkRequest req(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *reply = networkManager->get(req);
        connect(reply, &QNetworkReply::finished, this, [this, reply, fileName]() {
            const bool ok = (reply->error() == QNetworkReply::NoError);
            reply->deleteLater();
            if (ok) requestPrediction(fileName);
            else    startBackendAndWait([this, fileName]() { requestPrediction(fileName); });
        });
    }

    void startBackendAndWait(const std::function<void()> &onReady) {
        if (backendProcess->state() == QProcess::NotRunning) {
            const QString root = detectProjectRoot();
            const QString venv = root + "/.venv/bin/python";
            backendProcess->setProgram(QFile::exists(venv) ? venv : QString("python3"));
            backendProcess->setArguments({"-m", "backend"});
            backendProcess->setWorkingDirectory(root);
            backendProcess->start();
            backendStartedByGui = true;
        }
        statusBar()->showMessage("Starting backend server…");
        waitForBackend(onReady, 20);
    }

    void waitForBackend(const std::function<void()> &onReady, int attemptsLeft) {
        if (attemptsLeft <= 0) {
            statusBar()->showMessage("Backend unavailable.");
            predictionBadge->setText("Backend Error");
            QMessageBox::warning(this, "Backend Error",
                "Backend did not become ready in time.\n"
                "Start it manually with:  python -m backend");
            return;
        }
        QNetworkRequest req(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *reply = networkManager->get(req);
        connect(reply, &QNetworkReply::finished, this, [this, reply, onReady, attemptsLeft]() {
            const bool ok = (reply->error() == QNetworkReply::NoError);
            reply->deleteLater();
            if (ok) onReady();
            else    QTimer::singleShot(500, this, [this, onReady, attemptsLeft]() {
                        waitForBackend(onReady, attemptsLeft - 1);
                    });
        });
    }

    void refreshBackendStatus() {
        QNetworkRequest req(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *reply = networkManager->get(req);
        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            if (reply->error() == QNetworkReply::NoError) {
                const QJsonObject obj = QJsonDocument::fromJson(reply->readAll()).object();
                const QString model   = obj.value("model_name").toString("Unknown");
                backendStatusLabel->setText("● Online — " + model);
                backendStatusLabel->setStyleSheet("color: #27ae60; font-weight: bold;");
            } else {
                backendStatusLabel->setText("● Offline");
                backendStatusLabel->setStyleSheet("color: #e74c3c; font-weight: bold;");
            }
            reply->deleteLater();
        });
    }

    void requestPrediction(const QString &fileName) {
        QFile *file = new QFile(fileName);
        if (!file->open(QIODevice::ReadOnly)) {
            statusBar()->showMessage("Error: could not open image file.");
            predictionBadge->setText("File Error");
            QMessageBox::warning(this, "File Error", "Unable to open selected image file.");
            file->deleteLater();
            return;
        }

        auto *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

        QHttpPart imgPart;
        imgPart.setHeader(QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\"image\"; filename=\"" + QFileInfo(fileName).fileName() + "\""));
        imgPart.setBodyDevice(file);
        file->setParent(multiPart);
        multiPart->append(imgPart);

        QHttpPart namePart;
        namePart.setHeader(QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\"patient_name\""));
        namePart.setBody(nameInput->text().toUtf8());
        multiPart->append(namePart);

        QHttpPart agePart;
        agePart.setHeader(QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\"patient_age\""));
        agePart.setBody(QString::number(ageInput->value()).toUtf8());
        multiPart->append(agePart);

        QNetworkRequest request(QUrl("http://127.0.0.1:5000/predict"));
        QNetworkReply *reply = networkManager->post(request, multiPart);
        multiPart->setParent(reply);

        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            const QByteArray body = reply->readAll();
            if (reply->error() != QNetworkReply::NoError) {
                predictionBadge->setText("Request Failed");
                statusBar()->showMessage("Prediction request failed.");
                QString detail;
                const QJsonDocument errDoc = QJsonDocument::fromJson(body);
                if (errDoc.isObject()) detail = errDoc.object().value("error").toString();
                if (detail.isEmpty())  detail = reply->errorString();
                QMessageBox::warning(this, "Prediction Failed",
                    QString("Could not get prediction.\n\nDetails: %1").arg(detail));
                reply->deleteLater();
                return;
            }
            QJsonParseError parseErr;
            const QJsonDocument doc = QJsonDocument::fromJson(body, &parseErr);
            if (parseErr.error != QJsonParseError::NoError || !doc.isObject()) {
                predictionBadge->setText("Parse Error");
                statusBar()->showMessage("Invalid response from backend.");
                QMessageBox::warning(this, "Response Error", "Backend returned invalid JSON.");
                reply->deleteLater();
                return;
            }
            const QJsonObject obj = doc.object();
            if (obj.contains("error")) {
                predictionBadge->setText("Backend Error");
                statusBar()->showMessage("Backend error: " + obj.value("error").toString());
                QMessageBox::warning(this, "Prediction Error", obj.value("error").toString());
                reply->deleteLater();
                return;
            }
            updateAnalysisResults(obj);
            reply->deleteLater();
            refreshPatientRecords();   // auto-refresh the records tab
        });
    }

    void updateAnalysisResults(const QJsonObject &obj) {
        const QString prediction = obj.value("prediction").toString("Unknown");
        const double  confidence = obj.value("confidence").toDouble(0.0);
        const QString modelName  = obj.value("model_name").toString("Unknown");

        // ── Prediction badge ──────────────────────────────────────────────
        predictionBadge->setText(prediction);
        if (prediction == "AMD") {
            predictionBadge->setStyleSheet(
                "QLabel { background-color: #e74c3c; color: white; "
                "border-radius: 8px; padding: 8px 20px; font-size: 16px; font-weight: bold; }");
        } else if (prediction == "Normal") {
            predictionBadge->setStyleSheet(
                "QLabel { background-color: #27ae60; color: white; "
                "border-radius: 8px; padding: 8px 20px; font-size: 16px; font-weight: bold; }");
        } else {
            predictionBadge->setStyleSheet(
                "QLabel { background-color: #7f8c8d; color: white; "
                "border-radius: 8px; padding: 8px 20px; font-size: 16px; font-weight: bold; }");
        }

        // ── Risk level ────────────────────────────────────────────────────
        QString risk, riskStyle;
        if (prediction == "AMD") {
            if (confidence >= 0.85) {
                risk      = "⚠  High Risk";
                riskStyle = "QLabel { background-color:#c0392b; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            } else if (confidence >= 0.60) {
                risk      = "⚡  Moderate Risk";
                riskStyle = "QLabel { background-color:#e67e22; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            } else {
                risk      = "○  Low Risk";
                riskStyle = "QLabel { background-color:#f39c12; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            }
        } else {
            risk      = "✓  Healthy";
            riskStyle = "QLabel { background-color:#27ae60; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
        }
        riskLabel->setText(risk);
        riskLabel->setStyleSheet(riskStyle);

        // ── Confidence bar ────────────────────────────────────────────────
        const int confPct = static_cast<int>(confidence * 100.0);
        confidenceBar->setValue(confPct);
        confidenceValueLabel->setText(QString::number(confidence * 100.0, 'f', 1) + "%");
        const QString barColor = confidence >= 0.70 ? "#27ae60"
                               : confidence >= 0.50 ? "#f39c12"
                               :                      "#e74c3c";
        confidenceBar->setStyleSheet(
            QString("QProgressBar::chunk { background-color:%1; border-radius:4px; }").arg(barColor));

        // ── Metrics ───────────────────────────────────────────────────────
        auto fmt = [&obj](const char *key) -> QString {
            const QJsonValue v = obj.value(key);
            return v.isDouble() ? QString::number(v.toDouble() * 100.0, 'f', 1) + "%" : "N/A";
        };
        modelInfoLabel->setText(modelName);
        accuracyLabel ->setText(fmt("accuracy"));
        precisionLabel->setText(fmt("precision"));
        recallLabel   ->setText(fmt("recall"));
        f1Label       ->setText(fmt("f1_score"));

        // ── CAM image ─────────────────────────────────────────────────────
        const QString camPath = obj.value("cam_image_path").toString();
        if (!camPath.isEmpty() && QFile::exists(camPath))
            camsLabel->setPixmap(QPixmap(camPath));
        else
            camsLabel->setText("Saliency map\nnot available");

        statusBar()->showMessage(
            QString("Analysis complete — %1  (confidence: %2%)")
                .arg(prediction)
                .arg(QString::number(confidence * 100.0, 'f', 1))
        );
    }

    // ── Patient records ──────────────────────────────────────────────────────

    void refreshPatientRecords() {
        QNetworkRequest req(QUrl("http://127.0.0.1:5000/patients"));
        QNetworkReply *reply = networkManager->get(req);
        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            if (reply->error() != QNetworkReply::NoError) {
                reply->deleteLater();
                return;
            }
            const QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
            reply->deleteLater();
            if (!doc.isObject()) return;

            const QJsonArray patients = doc.object().value("patients").toArray();
            allPatients.clear();
            for (const QJsonValue &v : patients)
                if (v.isObject()) allPatients.append(v.toObject());

            populateTable(patients);
        });
    }

    void filterRecords(const QString &query) {
        if (query.isEmpty()) {
            QJsonArray all;
            for (const QJsonObject &p : allPatients) all.append(p);
            populateTable(all);
            return;
        }
        QJsonArray filtered;
        for (const QJsonObject &p : allPatients)
            if (p.value("name").toString().contains(query, Qt::CaseInsensitive))
                filtered.append(p);
        populateTable(filtered);
    }

    void populateTable(const QJsonArray &patients) {
        patientTable->setSortingEnabled(false);
        patientTable->setRowCount(0);

        int amdCount = 0, normalCount = 0;

        for (const QJsonValue &v : patients) {
            if (!v.isObject()) continue;
            const QJsonObject p = v.toObject();
            const int row = patientTable->rowCount();
            patientTable->insertRow(row);

            auto cell = [&](int col, const QString &text) {
                patientTable->setItem(row, col, new QTableWidgetItem(text));
            };

            cell(0, QString::number(p.value("id").toInt()));
            cell(1, p.value("name").toString("—"));

            const int age = p.value("age").toInt(0);
            cell(2, age > 0 ? QString::number(age) : "—");

            const QString diag = p.value("prediction").toString("—");
            QTableWidgetItem *diagItem = new QTableWidgetItem(diag);
            if (diag == "AMD") {
                diagItem->setForeground(QColor("#e74c3c"));
                QFont f = diagItem->font(); f.setBold(true); diagItem->setFont(f);
                ++amdCount;
            } else if (diag == "Normal") {
                diagItem->setForeground(QColor("#27ae60"));
                QFont f = diagItem->font(); f.setBold(true); diagItem->setFont(f);
                ++normalCount;
            }
            patientTable->setItem(row, 3, diagItem);

            const double conf = p.value("confidence").toDouble(0.0);
            cell(4, conf > 0.0 ? QString::number(conf * 100.0, 'f', 1) + "%" : "—");
            cell(5, p.value("date").toString("—"));

            // Store image path in the ID cell for click-to-load
            patientTable->item(row, 0)->setData(Qt::UserRole, p.value("image_path").toString());
        }

        patientTable->setSortingEnabled(true);

        // Update stats bar and sidebar scan count
        const int total = patients.size();
        recordsStatsLabel->setText(
            QString("Total: %1 scan(s)   |   AMD: %2   |   Normal: %3")
                .arg(total).arg(amdCount).arg(normalCount)
        );
        scanCountLabel->setText(QString("Total scans: %1").arg(total));
    }

    void onRecordSelected(int row, int) {
        QTableWidgetItem *idItem = patientTable->item(row, 0);
        if (!idItem) return;
        const QString imgPath = idItem->data(Qt::UserRole).toString();
        if (!imgPath.isEmpty() && QFile::exists(imgPath)) {
            fundusLabel->setPixmap(QPixmap(imgPath));
            tabWidget->setCurrentIndex(0);   // switch to Analysis tab
            statusBar()->showMessage(
                "Loaded historical image for: " + patientTable->item(row, 1)->text()
            );
        }
    }

    // ── Theme ────────────────────────────────────────────────────────────────

    void toggleTheme() {
        isDarkMode = !isDarkMode;
        settings->setValue("darkMode", isDarkMode);
        if (isDarkMode) applyDarkMode();
        else            applyLightMode();
    }

    void applyLightMode() {
        themeBtn->setText("🌙   Dark Mode");
        qApp->setStyleSheet(R"(
            /* ── Base ── */
            QMainWindow, QWidget {
                background-color: #f0f2f5;
                color: #2c3e50;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            /* ── Sidebar ── */
            QWidget#sidebar {
                background-color: #1e2d3d;
            }
            QWidget#sidebar QLabel {
                color: #ecf0f1;
            }
            QLabel#appTitle {
                color: #74b9ff;
                font-size: 20px;
                font-weight: bold;
            }
            QLabel#appSubtitle {
                color: #7f8c8d;
                font-size: 11px;
            }
            QLabel#sectionHeader {
                color: #95a5a6;
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#statusLabel, QLabel#infoLabel {
                color: #bdc3c7;
                font-size: 12px;
            }
            QGroupBox#patientGroup {
                color: #bdc3c7;
                border: 1px solid #2d4860;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox#patientGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #74b9ff;
                font-size: 12px;
            }
            QGroupBox#patientGroup QLabel {
                color: #bdc3c7;
            }
            QLineEdit#sidebarInput, QSpinBox#sidebarInput {
                background-color: #2d4860;
                color: #ecf0f1;
                border: 1px solid #3d6080;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QPushButton#primaryBtn {
                background-color: #2980b9;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#primaryBtn:hover   { background-color: #3498db; }
            QPushButton#primaryBtn:pressed { background-color: #1f6594; }
            QPushButton#secondaryBtn {
                background-color: #2d4860;
                color: #ecf0f1;
                border: 1px solid #3d6080;
                border-radius: 5px;
                padding: 5px 12px;
            }
            QPushButton#secondaryBtn:hover { background-color: #3d6080; }
            QFrame#separator  { color: #2d4860; }
            QFrame#vSeparator { color: #ced4da; }
            /* ── Tabs ── */
            QTabWidget#mainTabs::pane {
                border: none;
                background-color: #f0f2f5;
            }
            QTabBar::tab {
                background-color: #dde3ea;
                color: #5a6678;
                border: none;
                padding: 9px 22px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #f0f2f5;
                color: #2980b9;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected { background-color: #cfd6de; }
            /* ── Image groups ── */
            QGroupBox#imageGroup {
                background-color: white;
                border: 1px solid #dce2ea;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox#imageGroup::title { color: #7f8c8d; }
            QLabel#imageDisplay {
                background-color: #eef1f5;
                border-radius: 4px;
                color: #95a5a6;
            }
            /* ── Results group ── */
            QGroupBox#resultsGroup {
                background-color: white;
                border: 1px solid #dce2ea;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
            }
            QLabel#predictionBadge {
                background-color: #95a5a6;
                color: white;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#riskLabel {
                background-color: #95a5a6;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QLabel#metricHeader { color: #7f8c8d; }
            QLabel#metricValue  { color: #2980b9; font-weight: bold; }
            QProgressBar#confidenceBar {
                background-color: #eef1f5;
                border: 1px solid #dce2ea;
                border-radius: 4px;
            }
            QProgressBar#confidenceBar::chunk {
                background-color: #27ae60;
                border-radius: 4px;
            }
            /* ── Records tab ── */
            QLineEdit#searchBar {
                background-color: white;
                color: #2c3e50;
                border: 1px solid #dce2ea;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLabel#statsLabel {
                color: #7f8c8d;
                font-size: 12px;
                padding: 2px 0;
            }
            QTableWidget#patientTable {
                background-color: white;
                alternate-background-color: #f5f7fb;
                gridline-color: transparent;
                border: 1px solid #dce2ea;
                border-radius: 6px;
                selection-background-color: #d0e8fb;
                selection-color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #eaf2fb;
                color: #2c3e50;
                border: none;
                border-bottom: 2px solid #2980b9;
                padding: 7px 10px;
                font-weight: bold;
            }
            /* ── Status bar ── */
            QStatusBar {
                background-color: #1e2d3d;
                color: #74b9ff;
                font-size: 12px;
                padding-left: 6px;
            }
            /* ── About tab ── */
            QLabel#aboutTitle {
                font-size: 22px;
                font-weight: bold;
                color: #2980b9;
            }
            QLabel#aboutText { color: #2c3e50; line-height: 1.5; }
        )");
    }

    void applyDarkMode() {
        themeBtn->setText("☀   Light Mode");
        qApp->setStyleSheet(R"(
            /* ── Base ── */
            QMainWindow, QWidget {
                background-color: #1a1d24;
                color: #dfe6e9;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            /* ── Sidebar ── */
            QWidget#sidebar {
                background-color: #12151c;
            }
            QWidget#sidebar QLabel {
                color: #dfe6e9;
            }
            QLabel#appTitle {
                color: #74b9ff;
                font-size: 20px;
                font-weight: bold;
            }
            QLabel#appSubtitle {
                color: #636e72;
                font-size: 11px;
            }
            QLabel#sectionHeader {
                color: #74b9ff;
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#statusLabel, QLabel#infoLabel {
                color: #b2bec3;
                font-size: 12px;
            }
            QGroupBox#patientGroup {
                color: #dfe6e9;
                border: 1px solid #2d3436;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox#patientGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #74b9ff;
                font-size: 12px;
            }
            QGroupBox#patientGroup QLabel {
                color: #b2bec3;
            }
            QLineEdit#sidebarInput, QSpinBox#sidebarInput {
                background-color: #2d3436;
                color: #dfe6e9;
                border: 1px solid #636e72;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QPushButton#primaryBtn {
                background-color: #0984e3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#primaryBtn:hover   { background-color: #74b9ff; color: #12151c; }
            QPushButton#primaryBtn:pressed { background-color: #0773c5; }
            QPushButton#secondaryBtn {
                background-color: #2d3436;
                color: #dfe6e9;
                border: 1px solid #636e72;
                border-radius: 5px;
                padding: 5px 12px;
            }
            QPushButton#secondaryBtn:hover { background-color: #3d4446; }
            QFrame#separator  { color: #2d3436; }
            QFrame#vSeparator { color: #2d3436; }
            /* ── Tabs ── */
            QTabWidget#mainTabs::pane {
                border: none;
                background-color: #1a1d24;
            }
            QTabBar::tab {
                background-color: #2d3436;
                color: #b2bec3;
                border: none;
                padding: 9px 22px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #1a1d24;
                color: #74b9ff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected { background-color: #353b3e; }
            /* ── Image groups ── */
            QGroupBox#imageGroup {
                background-color: #232830;
                border: 1px solid #2d3436;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox#imageGroup::title { color: #74b9ff; }
            QLabel#imageDisplay {
                background-color: #2d3436;
                border-radius: 4px;
                color: #636e72;
            }
            /* ── Results group ── */
            QGroupBox#resultsGroup {
                background-color: #232830;
                border: 1px solid #2d3436;
                border-radius: 8px;
                margin-top: 10px;
            }
            QLabel#predictionBadge {
                background-color: #636e72;
                color: white;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#riskLabel {
                background-color: #636e72;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QLabel#metricHeader { color: #636e72; }
            QLabel#metricValue  { color: #74b9ff; font-weight: bold; }
            QProgressBar#confidenceBar {
                background-color: #2d3436;
                border: 1px solid #636e72;
                border-radius: 4px;
            }
            QProgressBar#confidenceBar::chunk {
                background-color: #0984e3;
                border-radius: 4px;
            }
            /* ── Records tab ── */
            QLineEdit#searchBar {
                background-color: #2d3436;
                color: #dfe6e9;
                border: 1px solid #636e72;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLabel#statsLabel {
                color: #636e72;
                font-size: 12px;
                padding: 2px 0;
            }
            QTableWidget#patientTable {
                background-color: #232830;
                alternate-background-color: #1e2228;
                gridline-color: transparent;
                border: 1px solid #2d3436;
                border-radius: 6px;
                selection-background-color: #2d4a6a;
                selection-color: #dfe6e9;
                color: #dfe6e9;
            }
            QHeaderView::section {
                background-color: #2d3436;
                color: #74b9ff;
                border: none;
                border-bottom: 2px solid #0984e3;
                padding: 7px 10px;
                font-weight: bold;
            }
            /* ── Status bar ── */
            QStatusBar {
                background-color: #12151c;
                color: #74b9ff;
                font-size: 12px;
                padding-left: 6px;
            }
            /* ── About tab ── */
            QLabel#aboutTitle {
                font-size: 22px;
                font-weight: bold;
                color: #74b9ff;
            }
            QLabel#aboutText { color: #dfe6e9; line-height: 1.5; }
        )");
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    QString detectProjectRoot() const {
        QDir dir(QCoreApplication::applicationDirPath());
        if (dir.cdUp() && dir.cdUp() && dir.exists("backend"))
            return dir.absolutePath();
        QDir cwd(QDir::currentPath());
        if (cwd.exists("backend"))
            return cwd.absolutePath();
        return QCoreApplication::applicationDirPath();
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    AMD_GUI window;
    window.show();
    return app.exec();
}

