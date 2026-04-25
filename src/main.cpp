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
#include <QDialog>
#include <QScrollArea>
#include <QTextStream>
#include <QDateTime>
#include <QMenu>
#include <QToolButton>
#include <QShortcut>
#include <QKeySequence>
#include <functional>

#ifndef AMD_PROJECT_SOURCE_DIR
#define AMD_PROJECT_SOURCE_DIR ""
#endif

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
    QLabel       *specificityLabel;
    QLabel       *f1Label;

    // ── Records tab ─────────────────────────────────────────────────────────
    QTableWidget *patientTable;
    QLineEdit    *searchInput;
    QLabel       *recordsStatsLabel;
    QLabel       *statTotalValue;
    QLabel       *statAMDValue;
    QLabel       *statNormalValue;
    QLabel       *statAvgConfValue;
    QPushButton  *deleteBtn;
    QPushButton  *exportBtn;
    QLabel       *recordsEmptyLabel;

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
        isDarkMode          = settings->value("darkMode", true).toBool();
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
        sideLayout->setContentsMargins(20, 24, 20, 24);
        sideLayout->setSpacing(14);

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
        ageInput->setRange(1, 120);
        ageInput->setValue(50);
        ageInput->setObjectName("sidebarInput");
        form->addRow("Age:", ageInput);

        sideLayout->addWidget(patientGroup);

        // Primary upload/analyse button
        uploadBtn = new QPushButton("Upload and Analyse");
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

        backendStatusLabel = new QLabel("Checking backend...");
        backendStatusLabel->setObjectName("statusLabel");
        backendStatusLabel->setWordWrap(true);
        sideLayout->addWidget(backendStatusLabel);

        scanCountLabel = new QLabel("Total scans: 0");
        scanCountLabel->setObjectName("infoLabel");
        sideLayout->addWidget(scanCountLabel);

        sideLayout->addStretch();

        // Theme toggle
        themeBtn = new QPushButton(isDarkMode ? "Light Mode" : "Dark Mode");
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
        addMetricPair(1, 1, "Recall (Sensitivity):", recallLabel);
        addMetricPair(2, 0, "Specificity:", specificityLabel);
        addMetricPair(2, 1, "F1 Score:",  f1Label);
        resultsLayout->addLayout(metricsGrid);

        analysisLayout->addWidget(resultsGrp);
        tabWidget->addTab(analysisTab, " Analysis ");

        // ── TAB 2: History ─────────────────────────────────────────
        QWidget     *recordsTab    = new QWidget();
        QVBoxLayout *recordsLayout = new QVBoxLayout(recordsTab);
        recordsLayout->setContentsMargins(16, 16, 16, 16);
        recordsLayout->setSpacing(12);

        // Stat tile row: Total / AMD / Normal / Avg confidence
        auto makeStatTile = [&](const QString &caption, QLabel *&valueLabel, const QString &accentObj) {
            QFrame *tile = new QFrame();
            tile->setObjectName("statTile");
            tile->setFrameShape(QFrame::NoFrame);
            QVBoxLayout *tl = new QVBoxLayout(tile);
            tl->setContentsMargins(16, 12, 16, 12);
            tl->setSpacing(4);

            QLabel *cap = new QLabel(caption);
            cap->setObjectName("statCaption");
            tl->addWidget(cap);

            valueLabel = new QLabel("0");
            valueLabel->setObjectName(accentObj);
            tl->addWidget(valueLabel);
            return tile;
        };

        QHBoxLayout *statsRow = new QHBoxLayout();
        statsRow->setSpacing(12);
        statsRow->addWidget(makeStatTile("TOTAL SCANS",     statTotalValue,   "statValue"));
        statsRow->addWidget(makeStatTile("AMD DETECTED",    statAMDValue,     "statValueAMD"));
        statsRow->addWidget(makeStatTile("NORMAL",          statNormalValue,  "statValueNormal"));
        statsRow->addWidget(makeStatTile("AVG CONFIDENCE",  statAvgConfValue, "statValue"));
        recordsLayout->addLayout(statsRow);

        // Toolbar row: search + actions
        QHBoxLayout *ctrlRow = new QHBoxLayout();
        ctrlRow->setSpacing(8);

        searchInput = new QLineEdit();
        searchInput->setPlaceholderText("Search by patient name, ID, or diagnosis…");
        searchInput->setObjectName("searchBar");
        searchInput->setClearButtonEnabled(true);
        ctrlRow->addWidget(searchInput, 1);

        QPushButton *refreshBtn = new QPushButton("Refresh");
        refreshBtn->setObjectName("secondaryBtn");
        refreshBtn->setMinimumHeight(36);
        refreshBtn->setMinimumWidth(96);
        ctrlRow->addWidget(refreshBtn);

        exportBtn = new QPushButton("Export CSV");
        exportBtn->setObjectName("secondaryBtn");
        exportBtn->setMinimumHeight(36);
        exportBtn->setMinimumWidth(110);
        ctrlRow->addWidget(exportBtn);

        deleteBtn = new QPushButton("Delete Selected");
        deleteBtn->setObjectName("dangerBtn");
        deleteBtn->setMinimumHeight(36);
        deleteBtn->setMinimumWidth(140);
        deleteBtn->setEnabled(false);
        ctrlRow->addWidget(deleteBtn);

        recordsLayout->addLayout(ctrlRow);

        // Stats summary label (text breakdown)
        recordsStatsLabel = new QLabel("No records yet — run a scan from the Analysis tab to populate this list.");
        recordsStatsLabel->setObjectName("statsLabel");
        recordsLayout->addWidget(recordsStatsLabel);

        // Stack the table + an empty-state hint inside a frame
        QFrame *tableFrame = new QFrame();
        tableFrame->setObjectName("tableFrame");
        QVBoxLayout *tableLayout = new QVBoxLayout(tableFrame);
        tableLayout->setContentsMargins(0, 0, 0, 0);
        tableLayout->setSpacing(0);

        // Patient table
        patientTable = new QTableWidget();
        patientTable->setObjectName("patientTable");
        patientTable->setColumnCount(7);
        patientTable->setHorizontalHeaderLabels(
            {"ID", "Patient Name", "Age", "Diagnosis", "Confidence", "Date", ""}
        );
        patientTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        patientTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        patientTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        patientTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        patientTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Fixed);
        patientTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
        patientTable->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Fixed);
        patientTable->horizontalHeader()->setDefaultSectionSize(120);
        patientTable->setColumnWidth(4, 170);
        patientTable->setColumnWidth(6, 50);
        patientTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        patientTable->setSelectionMode(QAbstractItemView::SingleSelection);
        patientTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        patientTable->setSortingEnabled(true);
        patientTable->verticalHeader()->setVisible(false);
        patientTable->verticalHeader()->setDefaultSectionSize(40);
        patientTable->setAlternatingRowColors(true);
        patientTable->setShowGrid(false);
        patientTable->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
        patientTable->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
        patientTable->setFocusPolicy(Qt::StrongFocus);
        patientTable->setContextMenuPolicy(Qt::CustomContextMenu);
        tableLayout->addWidget(patientTable);

        // Empty-state overlay (shown when there are zero records)
        recordsEmptyLabel = new QLabel("No patient records yet.\nUpload a fundus image from the Analysis tab to create your first record.");
        recordsEmptyLabel->setObjectName("emptyState");
        recordsEmptyLabel->setAlignment(Qt::AlignCenter);
        recordsEmptyLabel->setWordWrap(true);
        recordsEmptyLabel->setVisible(false);
        tableLayout->addWidget(recordsEmptyLabel);

        recordsLayout->addWidget(tableFrame, 1);

        tabWidget->addTab(recordsTab, " History ");

        root->addWidget(tabWidget, 1);

        // Status bar
        statusBar()->showMessage("Ready — upload a fundus image to begin analysis.");

        // ── CONNECTIONS ──────────────────────────────────────────────────
        connect(uploadBtn,  &QPushButton::clicked,       this, &AMD_GUI::uploadImage);
        connect(themeBtn,   &QPushButton::clicked,       this, &AMD_GUI::toggleTheme);
        connect(refreshBtn, &QPushButton::clicked,       this, &AMD_GUI::refreshPatientRecords);
        connect(exportBtn,  &QPushButton::clicked,       this, &AMD_GUI::exportRecordsCsv);
        connect(deleteBtn,  &QPushButton::clicked,       this, &AMD_GUI::deleteSelectedRecord);
        connect(searchInput, &QLineEdit::textChanged,    this, &AMD_GUI::filterRecords);
        connect(patientTable, &QTableWidget::cellClicked, this, &AMD_GUI::onRecordSelected);
        connect(patientTable, &QTableWidget::cellDoubleClicked,
                this, &AMD_GUI::onRecordDoubleClicked);
        connect(patientTable, &QTableWidget::itemSelectionChanged, this, [this]() {
            deleteBtn->setEnabled(!patientTable->selectedItems().isEmpty());
        });
        connect(patientTable, &QTableWidget::customContextMenuRequested,
                this, &AMD_GUI::showRecordContextMenu);

        // Keyboard shortcuts on the records tab
        auto *delShortcut = new QShortcut(QKeySequence(Qt::Key_Delete), patientTable);
        connect(delShortcut, &QShortcut::activated, this, &AMD_GUI::deleteSelectedRecord);
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
            QStringList backendArgs;
            backendProcess->setProgram(detectPythonExecutable(root, backendArgs));
            backendProcess->setArguments(backendArgs);
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
                backendStatusLabel->setText("Online  |  " + model);
                backendStatusLabel->setStyleSheet("color: #2ecc71; font-weight: 600;");
            } else {
                backendStatusLabel->setText("Offline");
                backendStatusLabel->setStyleSheet("color: #e74c3c; font-weight: 600;");
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
                bool invalidFundus = false;
                const QJsonDocument errDoc = QJsonDocument::fromJson(body);
                if (errDoc.isObject()) {
                    const QJsonObject errObj = errDoc.object();
                    detail = errObj.value("error").toString();
                    invalidFundus = errObj.value("invalid_fundus").toBool(false);
                }
                if (detail.isEmpty())  detail = reply->errorString();
                if (invalidFundus) {
                    predictionBadge->setText("Invalid Fundus Image");
                    predictionBadge->setStyleSheet(
                        "QLabel { background-color: #c0392b; color: white; "
                        "border-radius: 8px; padding: 8px 20px; font-size: 16px; font-weight: bold; }");
                    riskLabel->setText("Risk: N/A");
                    confidenceBar->setValue(0);
                    confidenceValueLabel->setText("N/A");
                    camsLabel->setText("No saliency map\nfor invalid fundus image");
                    statusBar()->showMessage("Invalid fundus image selected.");
                }
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
                risk      = "High Risk";
                riskStyle = "QLabel { background-color:#c0392b; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            } else if (confidence >= 0.60) {
                risk      = "Moderate Risk";
                riskStyle = "QLabel { background-color:#e67e22; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            } else {
                risk      = "Low Risk";
                riskStyle = "QLabel { background-color:#f39c12; color:white; border-radius:8px; padding:8px 16px; font-weight:bold; }";
            }
        } else {
            risk      = "Healthy";
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
        specificityLabel->setText(fmt("specificity"));
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
        patientTable->clearContents();
        patientTable->setRowCount(0);

        int amdCount = 0, normalCount = 0;
        double confSum = 0.0;
        int    confN   = 0;

        // Theme-aware diagnosis swatches: chosen to be readable on both
        // pitch-black and bright-white backgrounds.
        const QColor amdColor("#ef4444");      // red-500
        const QColor normalColor("#22c55e");   // green-500

        for (const QJsonValue &v : patients) {
            if (!v.isObject()) continue;
            const QJsonObject p = v.toObject();
            const int row = patientTable->rowCount();
            patientTable->insertRow(row);

            auto centeredItem = [&](const QString &text) {
                auto *it = new QTableWidgetItem(text);
                it->setTextAlignment(Qt::AlignCenter);
                return it;
            };
            auto leftItem = [&](const QString &text) {
                auto *it = new QTableWidgetItem(text);
                it->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
                return it;
            };

            const int patientId = p.value("id").toInt();
            QTableWidgetItem *idItem = centeredItem(QString::number(patientId));
            // Sort numerically by ID, not lexicographically.
            idItem->setData(Qt::DisplayRole, patientId);
            idItem->setData(Qt::UserRole, p.value("image_path").toString());
            patientTable->setItem(row, 0, idItem);

            patientTable->setItem(row, 1, leftItem(p.value("name").toString("Unknown")));

            const int age = p.value("age").toInt(0);
            patientTable->setItem(row, 2, centeredItem(age > 0 ? QString::number(age) : "—"));

            const QString diag = p.value("prediction").toString("—");
            QTableWidgetItem *diagItem = centeredItem(diag);
            QFont diagFont = diagItem->font();
            diagFont.setBold(true);
            diagFont.setLetterSpacing(QFont::AbsoluteSpacing, 0.5);
            diagItem->setFont(diagFont);
            if (diag == "AMD") {
                diagItem->setForeground(amdColor);
                ++amdCount;
            } else if (diag == "Normal") {
                diagItem->setForeground(normalColor);
                ++normalCount;
            }
            patientTable->setItem(row, 3, diagItem);

            const double conf = p.value("confidence").toDouble(0.0);
            QString confText = conf > 0.0
                ? QString::number(conf * 100.0, 'f', 1) + " %"
                : "—";
            QTableWidgetItem *confItem = centeredItem(confText);
            // Numeric sort key so the column sorts by value, not string.
            confItem->setData(Qt::UserRole + 1, conf);
            // Tint the confidence value to mirror the diagnosis colour, faded
            // when confidence is low so the user gets a quick visual sense.
            if (conf > 0.0) {
                const QColor base = (diag == "AMD") ? amdColor
                                  : (diag == "Normal") ? normalColor
                                  : QColor("#9ca3af");
                QColor tinted = base;
                tinted.setAlphaF(0.55 + 0.45 * std::min(1.0, std::max(0.0, conf)));
                confItem->setForeground(tinted);
                if (conf >= 0.85) {
                    QFont f = confItem->font(); f.setBold(true); confItem->setFont(f);
                }
                confSum += conf;
                ++confN;
            }
            patientTable->setItem(row, 4, confItem);

            patientTable->setItem(row, 5, centeredItem(p.value("date").toString("—")));

            // Final column: an "open image" affordance, only when we have a path.
            const QString imgPath = p.value("image_path").toString();
            QTableWidgetItem *openItem = centeredItem(imgPath.isEmpty() ? "" : "↗");
            openItem->setToolTip(imgPath.isEmpty() ? "" : "Double-click row or click ↗ to view full image");
            openItem->setData(Qt::UserRole, imgPath);
            patientTable->setItem(row, 6, openItem);
        }

        patientTable->setSortingEnabled(true);

        // Stat tiles & sidebar counter
        const int total = patients.size();
        statTotalValue->setText(QString::number(total));
        statAMDValue->setText(QString::number(amdCount));
        statNormalValue->setText(QString::number(normalCount));
        statAvgConfValue->setText(
            confN > 0 ? QString::number((confSum / confN) * 100.0, 'f', 1) + " %"
                      : "—"
        );

        recordsStatsLabel->setText(
            total == 0
                ? QString("No records yet — run a scan from the Analysis tab to populate this list.")
                : QString("Showing %1 record(s) — %2 AMD, %3 Normal.")
                      .arg(total).arg(amdCount).arg(normalCount)
        );
        scanCountLabel->setText(QString("Total scans: %1").arg(allPatients.size()));

        // Empty-state overlay
        const bool empty = (total == 0);
        recordsEmptyLabel->setVisible(empty);
        patientTable->setVisible(!empty);

        deleteBtn->setEnabled(!patientTable->selectedItems().isEmpty());
    }

    void onRecordSelected(int row, int col) {
        QTableWidgetItem *idItem = patientTable->item(row, 0);
        if (!idItem) return;
        const QString imgPath = idItem->data(Qt::UserRole).toString();

        // Clicking the "↗" cell opens the full-size image preview directly.
        if (col == 6 && !imgPath.isEmpty() && QFile::exists(imgPath)) {
            openFullImageDialog(imgPath, patientTable->item(row, 1)->text());
            return;
        }

        if (!imgPath.isEmpty() && QFile::exists(imgPath)) {
            fundusLabel->setPixmap(QPixmap(imgPath));
            statusBar()->showMessage(
                "Loaded historical image for: " + patientTable->item(row, 1)->text()
            );
        }
    }

    void onRecordDoubleClicked(int row, int /*col*/) {
        QTableWidgetItem *idItem = patientTable->item(row, 0);
        if (!idItem) return;
        const QString imgPath = idItem->data(Qt::UserRole).toString();
        if (imgPath.isEmpty() || !QFile::exists(imgPath)) {
            statusBar()->showMessage("No stored image found on disk for this record.");
            return;
        }
        openFullImageDialog(imgPath, patientTable->item(row, 1)->text());
    }

    void openFullImageDialog(const QString &imgPath, const QString &title) {
        QDialog *dlg = new QDialog(this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->setWindowTitle("Fundus Image — " + title);
        dlg->resize(820, 820);

        QVBoxLayout *lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(8, 8, 8, 8);

        QScrollArea *scroll = new QScrollArea(dlg);
        scroll->setWidgetResizable(true);
        QLabel *imgLbl = new QLabel();
        imgLbl->setAlignment(Qt::AlignCenter);
        QPixmap pix(imgPath);
        if (!pix.isNull())
            imgLbl->setPixmap(pix.scaled(800, 800, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        else
            imgLbl->setText("Failed to load image:\n" + imgPath);
        scroll->setWidget(imgLbl);
        lay->addWidget(scroll, 1);

        QPushButton *closeBtn = new QPushButton("Close");
        closeBtn->setObjectName("secondaryBtn");
        connect(closeBtn, &QPushButton::clicked, dlg, &QDialog::accept);
        lay->addWidget(closeBtn, 0, Qt::AlignRight);

        dlg->show();
    }

    void showRecordContextMenu(const QPoint &pos) {
        const int row = patientTable->rowAt(pos.y());
        if (row < 0) return;
        patientTable->selectRow(row);

        QMenu menu(patientTable);
        QAction *aLoad   = menu.addAction("Load image into Analysis tab");
        QAction *aOpen   = menu.addAction("Open full image…");
        menu.addSeparator();
        QAction *aDelete = menu.addAction("Delete record");

        QAction *chosen = menu.exec(patientTable->viewport()->mapToGlobal(pos));
        if (!chosen) return;

        QTableWidgetItem *idItem = patientTable->item(row, 0);
        const QString imgPath = idItem ? idItem->data(Qt::UserRole).toString() : QString();

        if (chosen == aLoad && !imgPath.isEmpty() && QFile::exists(imgPath)) {
            fundusLabel->setPixmap(QPixmap(imgPath));
            tabWidget->setCurrentIndex(0);
            statusBar()->showMessage("Loaded historical image for: " + patientTable->item(row, 1)->text());
        } else if (chosen == aOpen && !imgPath.isEmpty() && QFile::exists(imgPath)) {
            openFullImageDialog(imgPath, patientTable->item(row, 1)->text());
        } else if (chosen == aDelete) {
            deleteSelectedRecord();
        }
    }

    void deleteSelectedRecord() {
        const int row = patientTable->currentRow();
        if (row < 0) return;
        QTableWidgetItem *idItem = patientTable->item(row, 0);
        if (!idItem) return;
        const int patientId = idItem->data(Qt::DisplayRole).toInt();
        const QString name  = patientTable->item(row, 1) ? patientTable->item(row, 1)->text() : QString::number(patientId);

        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "Delete patient record",
            QString("Permanently delete the record for \"%1\" (ID %2)?\nThis cannot be undone.")
                .arg(name).arg(patientId),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No
        );
        if (reply != QMessageBox::Yes) return;

        QNetworkRequest req(QUrl(QString("http://127.0.0.1:5000/patients/%1").arg(patientId)));
        QNetworkReply *reply2 = networkManager->sendCustomRequest(req, "DELETE");
        connect(reply2, &QNetworkReply::finished, this, [this, reply2, name, patientId]() {
            const bool ok = (reply2->error() == QNetworkReply::NoError);
            reply2->deleteLater();
            if (ok) {
                statusBar()->showMessage(QString("Deleted record %1 (%2).").arg(patientId).arg(name));
                refreshPatientRecords();
            } else {
                QMessageBox::warning(this, "Delete failed",
                                     "Could not delete the record. Is the backend running?");
            }
        });
    }

    void exportRecordsCsv() {
        if (allPatients.isEmpty()) {
            QMessageBox::information(this, "Nothing to export", "There are no patient records to export yet.");
            return;
        }

        const QString defaultName = "amd_patient_records_" +
            QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".csv";
        const QString path = QFileDialog::getSaveFileName(
            this, "Export patient records", defaultName, "CSV files (*.csv)"
        );
        if (path.isEmpty()) return;

        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Export failed", "Could not open the file for writing.");
            return;
        }

        QTextStream out(&f);
        out << "id,name,age,diagnosis,confidence_pct,date,image_path\n";

        auto csvField = [](const QString &v) {
            QString s = v;
            s.replace('"', "\"\"");
            if (s.contains(',') || s.contains('\n') || s.contains('"'))
                s = '"' + s + '"';
            return s;
        };

        for (const QJsonObject &p : allPatients) {
            const QString conf = p.value("confidence").isDouble()
                ? QString::number(p.value("confidence").toDouble() * 100.0, 'f', 2)
                : QString();
            out << p.value("id").toInt()                       << ','
                << csvField(p.value("name").toString())        << ','
                << p.value("age").toInt()                      << ','
                << csvField(p.value("prediction").toString())  << ','
                << conf                                        << ','
                << csvField(p.value("date").toString())        << ','
                << csvField(p.value("image_path").toString())  << '\n';
        }
        f.close();
        statusBar()->showMessage("Exported " + QString::number(allPatients.size()) + " record(s) to " + path);
    }

    // ── Theme ────────────────────────────────────────────────────────────────

    void toggleTheme() {
        isDarkMode = !isDarkMode;
        settings->setValue("darkMode", isDarkMode);
        if (isDarkMode) applyDarkMode();
        else            applyLightMode();
    }

    void applyLightMode() {
        themeBtn->setText("Dark Mode");
        qApp->setStyleSheet(R"(
            QMainWindow, QWidget {
                background-color: #f5f6f8;
                color: #1a1f2b;
                font-family: "Inter", "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            QWidget#sidebar {
                background-color: #ffffff;
                border-right: 1px solid #e3e6eb;
            }
            QWidget#sidebar QLabel { color: #1a1f2b; }
            QLabel#sectionHeader {
                color: #6b7280;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.5px;
                padding-top: 4px;
            }
            QLabel#statusLabel, QLabel#infoLabel {
                color: #4b5563;
                font-size: 12px;
            }
            QGroupBox#patientGroup {
                color: #6b7280;
                border: 1px solid #e3e6eb;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox#patientGroup::title {
                subcontrol-origin: margin;
                left: 12px;
                color: #1a1f2b;
                font-size: 12px;
            }
            QGroupBox#patientGroup QLabel { color: #4b5563; }
            QLineEdit#sidebarInput, QSpinBox#sidebarInput {
                background-color: #f5f6f8;
                color: #1a1f2b;
                border: 1px solid #e3e6eb;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QLineEdit#sidebarInput:focus, QSpinBox#sidebarInput:focus {
                border: 1px solid #1a1f2b;
            }
            QPushButton#primaryBtn {
                background-color: #1a1f2b;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton#primaryBtn:hover   { background-color: #2d3142; }
            QPushButton#primaryBtn:pressed { background-color: #0d1117; }
            QPushButton#secondaryBtn {
                background-color: #ffffff;
                color: #1a1f2b;
                border: 1px solid #e3e6eb;
                border-radius: 6px;
                padding: 7px 14px;
            }
            QPushButton#secondaryBtn:hover { background-color: #f5f6f8; }
            QFrame#separator  { color: #e3e6eb; }
            QFrame#vSeparator { color: #e3e6eb; }
            QTabWidget#mainTabs::pane {
                border: none;
                background-color: #f5f6f8;
            }
            QTabBar::tab {
                background-color: transparent;
                color: #6b7280;
                border: none;
                padding: 10px 20px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                color: #1a1f2b;
                border-bottom: 2px solid #1a1f2b;
            }
            QTabBar::tab:hover:!selected { color: #1a1f2b; }
            QGroupBox#imageGroup, QGroupBox#resultsGroup {
                background-color: #ffffff;
                border: 1px solid #e3e6eb;
                border-radius: 10px;
                margin-top: 12px;
                font-weight: 600;
                color: #6b7280;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #1a1f2b;
            }
            QLabel#imageDisplay {
                background-color: #f5f6f8;
                border-radius: 6px;
                color: #9ca3af;
            }
            QLabel#predictionBadge {
                background-color: #6b7280;
                color: white;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#riskLabel {
                background-color: #6b7280;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 700;
            }
            QLabel#metricHeader { color: #6b7280; font-weight: 500; }
            QLabel#metricValue  { color: #1a1f2b; font-weight: 700; }
            QProgressBar#confidenceBar {
                background-color: #f5f6f8;
                border: 1px solid #e3e6eb;
                border-radius: 6px;
            }
            QProgressBar#confidenceBar::chunk {
                background-color: #1a1f2b;
                border-radius: 6px;
            }
            QLineEdit#searchBar {
                background-color: #ffffff;
                color: #1a1f2b;
                border: 1px solid #e3e6eb;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QLineEdit#searchBar:focus { border: 1px solid #1a1f2b; }
            QLabel#statsLabel {
                color: #6b7280;
                font-size: 12px;
                padding: 2px 0;
            }
            QFrame#tableFrame {
                background-color: #ffffff;
                border: 1px solid #e3e6eb;
                border-radius: 10px;
            }
            QTableWidget#patientTable {
                background-color: #ffffff;
                alternate-background-color: #f4f6f9;
                gridline-color: transparent;
                border: none;
                border-radius: 10px;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
                color: #1a1f2b;
            }
            QTableWidget#patientTable::item {
                padding: 8px 10px;
                border: none;
            }
            QTableWidget#patientTable::item:selected {
                background-color: #dbeafe;
                color: #0f172a;
            }
            QTableWidget#patientTable::item:hover {
                background-color: #eef2f7;
            }
            QHeaderView::section {
                background-color: #ffffff;
                color: #1a1f2b;
                border: none;
                border-bottom: 1px solid #e3e6eb;
                padding: 11px 12px;
                font-weight: 700;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            QFrame#statTile {
                background-color: #ffffff;
                border: 1px solid #e3e6eb;
                border-radius: 10px;
            }
            QLabel#statCaption {
                color: #6b7280;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.4px;
            }
            QLabel#statValue {
                color: #0f172a;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#statValueAMD {
                color: #dc2626;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#statValueNormal {
                color: #16a34a;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#emptyState {
                color: #6b7280;
                font-size: 13px;
                padding: 60px 20px;
                background: transparent;
            }
            QPushButton#dangerBtn {
                background-color: #ffffff;
                color: #dc2626;
                border: 1px solid #fecaca;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton#dangerBtn:hover  { background-color: #fef2f2; }
            QPushButton#dangerBtn:pressed{ background-color: #fee2e2; }
            QPushButton#dangerBtn:disabled {
                color: #d1d5db; border-color: #e5e7eb; background-color: #f9fafb;
            }
            QStatusBar {
                background-color: #ffffff;
                color: #4b5563;
                font-size: 12px;
                padding-left: 10px;
                border-top: 1px solid #e3e6eb;
            }
        )");
    }

    void applyDarkMode() {
        themeBtn->setText("Light Mode");
        qApp->setStyleSheet(R"(
            QMainWindow, QWidget {
                background-color: #000000;
                color: #e5e7eb;
                font-family: "Inter", "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            QWidget#sidebar {
                background-color: #000000;
                border-right: 1px solid #1a1a1a;
            }
            QWidget#sidebar QLabel { color: #e5e7eb; }
            QLabel#sectionHeader {
                color: #6b7280;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.5px;
                padding-top: 4px;
            }
            QLabel#statusLabel, QLabel#infoLabel {
                color: #9ca3af;
                font-size: 12px;
            }
            QGroupBox#patientGroup {
                color: #6b7280;
                background-color: #000000;
                border: 1px solid #1a1a1a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox#patientGroup::title {
                subcontrol-origin: margin;
                left: 12px;
                color: #ffffff;
                font-size: 12px;
            }
            QGroupBox#patientGroup QLabel { color: #9ca3af; }
            QLineEdit#sidebarInput, QSpinBox#sidebarInput {
                background-color: #0a0a0a;
                color: #e5e7eb;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                padding: 7px 10px;
                selection-background-color: #ffffff;
                selection-color: #000000;
            }
            QLineEdit#sidebarInput:focus, QSpinBox#sidebarInput:focus {
                border: 1px solid #ffffff;
            }
            QPushButton#primaryBtn {
                background-color: #ffffff;
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton#primaryBtn:hover   { background-color: #d1d5db; }
            QPushButton#primaryBtn:pressed { background-color: #9ca3af; }
            QPushButton#secondaryBtn {
                background-color: #0a0a0a;
                color: #e5e7eb;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                padding: 7px 14px;
            }
            QPushButton#secondaryBtn:hover { background-color: #141414; }
            QFrame#separator  { color: #1a1a1a; }
            QFrame#vSeparator { color: #1a1a1a; }
            /* ── Tabs ── */
            QTabWidget#mainTabs::pane {
                border: none;
                background-color: #000000;
            }
            QTabBar::tab {
                background-color: transparent;
                color: #6b7280;
                border: none;
                padding: 10px 20px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                color: #ffffff;
                border-bottom: 2px solid #ffffff;
            }
            QTabBar::tab:hover:!selected { color: #e5e7eb; }
            QGroupBox#imageGroup, QGroupBox#resultsGroup {
                background-color: #000000;
                border: 1px solid #1a1a1a;
                border-radius: 10px;
                margin-top: 12px;
                font-weight: 600;
                color: #6b7280;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #ffffff;
            }
            QLabel#imageDisplay {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                color: #4b5563;
            }
            QLabel#predictionBadge {
                background-color: #1a1a1a;
                color: white;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#riskLabel {
                background-color: #1a1a1a;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 700;
            }
            QLabel#metricHeader { color: #6b7280; font-weight: 500; }
            QLabel#metricValue  { color: #ffffff; font-weight: 700; }
            QProgressBar#confidenceBar {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
            }
            QProgressBar#confidenceBar::chunk {
                background-color: #ffffff;
                border-radius: 6px;
            }
            QLineEdit#searchBar {
                background-color: #0a0a0a;
                color: #e5e7eb;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QLineEdit#searchBar:focus { border: 1px solid #ffffff; }
            QLabel#statsLabel {
                color: #6b7280;
                font-size: 12px;
                padding: 2px 0;
            }
            QFrame#tableFrame {
                background-color: #050505;
                border: 1px solid #1f1f1f;
                border-radius: 10px;
            }
            QTableWidget#patientTable {
                background-color: #050505;
                alternate-background-color: #111111;
                gridline-color: transparent;
                border: none;
                border-radius: 10px;
                selection-background-color: #1f2937;
                selection-color: #ffffff;
                color: #e5e7eb;
            }
            QTableWidget#patientTable::item {
                padding: 8px 10px;
                border: none;
                color: #e5e7eb;
            }
            QTableWidget#patientTable::item:selected {
                background-color: #1f2937;
                color: #ffffff;
            }
            QTableWidget#patientTable::item:hover {
                background-color: #161616;
            }
            QHeaderView::section {
                background-color: #050505;
                color: #ffffff;
                border: none;
                border-bottom: 1px solid #1f1f1f;
                padding: 11px 12px;
                font-weight: 700;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            QFrame#statTile {
                background-color: #0a0a0a;
                border: 1px solid #1f1f1f;
                border-radius: 10px;
            }
            QLabel#statCaption {
                color: #6b7280;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.4px;
            }
            QLabel#statValue {
                color: #ffffff;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#statValueAMD {
                color: #ef4444;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#statValueNormal {
                color: #22c55e;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#emptyState {
                color: #6b7280;
                font-size: 13px;
                padding: 60px 20px;
                background: transparent;
            }
            QPushButton#dangerBtn {
                background-color: #0a0a0a;
                color: #f87171;
                border: 1px solid #3f1d1d;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton#dangerBtn:hover  { background-color: #1a0e0e; border-color: #7f1d1d; }
            QPushButton#dangerBtn:pressed{ background-color: #2b0e0e; }
            QPushButton#dangerBtn:disabled {
                color: #4b5563; border-color: #1a1a1a; background-color: #050505;
            }
            QMenu {
                background-color: #0a0a0a;
                color: #e5e7eb;
                border: 1px solid #1f1f1f;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item { padding: 6px 18px; border-radius: 4px; }
            QMenu::item:selected { background-color: #1f2937; color: #ffffff; }
            QMenu::separator { height: 1px; background: #1f1f1f; margin: 4px 6px; }
            QStatusBar {
                background-color: #000000;
                color: #9ca3af;
                font-size: 12px;
                padding-left: 10px;
                border-top: 1px solid #1a1a1a;
            }
        )");
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    QString detectProjectRoot() const {
        QDir dir(QCoreApplication::applicationDirPath());
        for (int i = 0; i < 8; ++i) {
            if (dir.exists("backend"))
                return dir.absolutePath();
            if (!dir.cdUp())
                break;
        }

        QDir cwd(QDir::currentPath());
        if (cwd.exists("backend"))
            return cwd.absolutePath();

        QDir sourceDir(QStringLiteral(AMD_PROJECT_SOURCE_DIR));
        if (sourceDir.exists("backend"))
            return sourceDir.absolutePath();

        return QCoreApplication::applicationDirPath();
    }

    QString detectPythonExecutable(const QString &projectRoot, QStringList &arguments) const {
        arguments = QStringList({"-m", "backend"});

#ifdef Q_OS_WIN
        const QStringList candidates = {
            projectRoot + "/.venv/Scripts/python.exe",
            projectRoot + "/venv/Scripts/python.exe",
            projectRoot + "/.venv/Scripts/python",
            projectRoot + "/venv/Scripts/python"
        };
#else
        const QStringList candidates = {
            projectRoot + "/.venv/bin/python",
            projectRoot + "/venv/bin/python",
            projectRoot + "/.venv/bin/python3",
            projectRoot + "/venv/bin/python3"
        };
#endif

        for (const QString &candidate : candidates) {
            if (QFile::exists(candidate))
                return candidate;
        }

#ifdef Q_OS_WIN
        arguments = QStringList({"-3", "-m", "backend"});
        return "py";
#else
        return "python3";
#endif
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    AMD_GUI window;
    window.show();
    return app.exec();
}
