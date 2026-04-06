#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPixmap>
#include <QFrame>
#include <QListWidget>
#include <QListWidgetItem>
#include <QSettings>
#include <QFileInfo>
#include <QFile>
#include <QHttpMultiPart>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QCoreApplication>
#include <QDir>
#include <QTimer>
#include <QUrl>
#include <functional>

class AMD_GUI : public QWidget {
public:
    QLabel *fundusLabel;
    QLabel *camsLabel;
    QLabel *diagnosisLabel;
    QLabel *backendStatusLabel;
    QLineEdit *nameInput;
    QListWidget *historyList;
    QPushButton *themeBtn;
    bool isDarkMode;
    QSettings *settings;
    QNetworkAccessManager *networkManager;
    QProcess *backendProcess;
    bool backendStartedByGui;

    AMD_GUI() {
        setWindowTitle("AMD Detection System");
        resize(1000, 600);
        
        // Initialize settings for theme persistence
        settings = new QSettings("AMD_Detection", "AMD_GUI", this);
        isDarkMode = settings->value("darkMode", false).toBool();
        networkManager = new QNetworkAccessManager(this);
        backendProcess = new QProcess(this);
        backendStartedByGui = false;

        nameInput = new QLineEdit();
        nameInput->setPlaceholderText("Enter Patient Name");

        QPushButton *uploadBtn = new QPushButton("Upload Fundus Image");
        
        themeBtn = new QPushButton(isDarkMode ? "Light Mode" : "Dark Mode");
        themeBtn->setMaximumWidth(150);

        fundusLabel = new QLabel("Fundus Image");
        fundusLabel->setFrameShape(QFrame::Box);
        fundusLabel->setFixedSize(300,300);
        fundusLabel->setScaledContents(true);

        camsLabel = new QLabel("CAMS Image");
        camsLabel->setFrameShape(QFrame::Box);
        camsLabel->setFixedSize(300,300);
        camsLabel->setScaledContents(true);

        diagnosisLabel = new QLabel("Diagnosis: ");
        diagnosisLabel->setFrameShape(QFrame::Box);

        backendStatusLabel = new QLabel("Backend: Checking...");
        backendStatusLabel->setFrameShape(QFrame::Box);
        backendStatusLabel->setMinimumWidth(200);

        historyList = new QListWidget();
        historyList->setFixedWidth(200);

        QVBoxLayout *historyLayout = new QVBoxLayout();
        historyLayout->addWidget(new QLabel("Fundus History"));
        historyLayout->addWidget(historyList);

        QHBoxLayout *imgLayout = new QHBoxLayout();
        imgLayout->addWidget(fundusLabel);
        imgLayout->addWidget(camsLabel);
        imgLayout->addLayout(historyLayout);
        
        QHBoxLayout *topLayout = new QHBoxLayout();
        topLayout->addWidget(uploadBtn);
        topLayout->addWidget(backendStatusLabel);
        topLayout->addStretch();
        topLayout->addWidget(themeBtn);

        QVBoxLayout *mainLayout = new QVBoxLayout();
        mainLayout->addWidget(nameInput);
        mainLayout->addLayout(topLayout);
        mainLayout->addLayout(imgLayout);
        mainLayout->addWidget(diagnosisLabel);

        setLayout(mainLayout);

        // Apply initial theme
        if (isDarkMode) {
            applyDarkMode();
        } else {
            applyLightMode();
        }

        // Theme toggle button
        connect(themeBtn, &QPushButton::clicked, this, &AMD_GUI::toggleTheme);

        // Upload button functionality
        connect(uploadBtn, &QPushButton::clicked, this, [this](){
            QString fileName = QFileDialog::getOpenFileName(
                this,
                "Select Fundus Image",
                "",
                "Images (*.png *.jpg *.jpeg)"
            );

            if(!fileName.isEmpty()) {
                fundusLabel->setPixmap(QPixmap(fileName));
                historyList->addItem(fileName);
                diagnosisLabel->setText("Diagnosis: Processing...");
                ensureBackendAndPredict(fileName);
            }
        });

        // History selection functionality
        connect(historyList, &QListWidget::itemClicked, this, [this](QListWidgetItem *item){
            fundusLabel->setPixmap(QPixmap(item->text()));
        });

        auto *statusTimer = new QTimer(this);
        connect(statusTimer, &QTimer::timeout, this, [this]() {
            refreshBackendStatus();
        });
        statusTimer->start(3000);
        refreshBackendStatus();
    }

    ~AMD_GUI() override {
        if (backendStartedByGui && backendProcess->state() != QProcess::NotRunning) {
            backendProcess->terminate();
            if (!backendProcess->waitForFinished(1500)) {
                backendProcess->kill();
            }
        }
    }

    QString detectProjectRoot() const {
        QDir dir(QCoreApplication::applicationDirPath());
        if (dir.cdUp() && dir.cdUp() && dir.exists("backend")) {
            return dir.absolutePath();
        }

        QDir fallback(QDir::currentPath());
        if (fallback.exists("backend")) {
            return fallback.absolutePath();
        }

        return QCoreApplication::applicationDirPath();
    }

    void ensureBackendAndPredict(const QString &fileName) {
        QNetworkRequest healthReq(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *healthReply = networkManager->get(healthReq);

        connect(healthReply, &QNetworkReply::finished, this, [this, healthReply, fileName]() {
            const bool backendReady = (healthReply->error() == QNetworkReply::NoError);
            healthReply->deleteLater();

            if (backendReady) {
                requestPrediction(fileName);
                return;
            }

            startBackendAndWait([this, fileName]() {
                requestPrediction(fileName);
            });
        });
    }

    void startBackendAndWait(const std::function<void()> &onReady) {
        if (backendProcess->state() == QProcess::NotRunning) {
            const QString projectRoot = detectProjectRoot();
            const QString venvPython = projectRoot + "/.venv/bin/python";

            QString program = QFile::exists(venvPython) ? venvPython : QString("python3");
            QStringList args;
            args << "-m" << "backend";

            backendProcess->setProgram(program);
            backendProcess->setArguments(args);
            backendProcess->setWorkingDirectory(projectRoot);
            backendProcess->start();
            backendStartedByGui = true;
        }

        diagnosisLabel->setText("Diagnosis: Starting backend...");
        waitForBackend(onReady, 20);
    }

    void waitForBackend(const std::function<void()> &onReady, int attemptsLeft) {
        if (attemptsLeft <= 0) {
            diagnosisLabel->setText("Diagnosis: Backend unavailable");
            QMessageBox::warning(
                this,
                "Backend Error",
                "Backend did not become ready in time. Please start it manually with: python -m backend"
            );
            return;
        }

        QNetworkRequest healthReq(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *healthReply = networkManager->get(healthReq);

        connect(healthReply, &QNetworkReply::finished, this, [this, healthReply, onReady, attemptsLeft]() {
            const bool backendReady = (healthReply->error() == QNetworkReply::NoError);
            healthReply->deleteLater();

            if (backendReady) {
                onReady();
                return;
            }

            QTimer::singleShot(500, this, [this, onReady, attemptsLeft]() {
                waitForBackend(onReady, attemptsLeft - 1);
            });
        });
    }

    void refreshBackendStatus() {
        QNetworkRequest healthReq(QUrl("http://127.0.0.1:5000/health"));
        QNetworkReply *reply = networkManager->get(healthReq);

        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            if (reply->error() == QNetworkReply::NoError) {
                const QJsonDocument healthDoc = QJsonDocument::fromJson(reply->readAll());
                const QString modelType = healthDoc.object().value("model_type").toString("unknown");
                backendStatusLabel->setText(QString("Backend: Online (%1)").arg(modelType));
                backendStatusLabel->setStyleSheet("QLabel { color: #2e7d32; }");
            } else {
                backendStatusLabel->setText("Backend: Offline");
                backendStatusLabel->setStyleSheet("QLabel { color: #c62828; }");
            }
            reply->deleteLater();
        });
    }

    void requestPrediction(const QString &fileName) {
        QFile *file = new QFile(fileName);
        if (!file->open(QIODevice::ReadOnly)) {
            diagnosisLabel->setText("Diagnosis: Failed to open image file");
            QMessageBox::warning(this, "File Error", "Unable to open selected image file.");
            file->deleteLater();
            return;
        }

        auto *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

        QHttpPart imagePart;
        imagePart.setHeader(
            QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\"image\"; filename=\"" + QFileInfo(fileName).fileName() + "\"")
        );
        imagePart.setBodyDevice(file);
        file->setParent(multiPart);
        multiPart->append(imagePart);

        QHttpPart patientPart;
        patientPart.setHeader(
            QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\"patient_name\"")
        );
        patientPart.setBody(nameInput->text().toUtf8());
        multiPart->append(patientPart);

        QNetworkRequest request(QUrl("http://127.0.0.1:5000/predict"));
        QNetworkReply *reply = networkManager->post(request, multiPart);
        multiPart->setParent(reply);

        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            const QByteArray responseBody = reply->readAll();

            if (reply->error() != QNetworkReply::NoError) {
                diagnosisLabel->setText("Diagnosis: Request failed");

                QString backendDetail;
                QJsonParseError err;
                const QJsonDocument errorDoc = QJsonDocument::fromJson(responseBody, &err);
                if (err.error == QJsonParseError::NoError && errorDoc.isObject()) {
                    backendDetail = errorDoc.object().value("error").toString();
                }

                const QString details = backendDetail.isEmpty()
                    ? reply->errorString()
                    : backendDetail;

                QMessageBox::warning(
                    this,
                    "Backend Error",
                    "Could not reach backend at http://127.0.0.1:5000.\n\n"
                    "Details: " + details
                );
                reply->deleteLater();
                return;
            }

            QJsonParseError parseError;
            const QJsonDocument jsonDoc = QJsonDocument::fromJson(responseBody, &parseError);
            if (parseError.error != QJsonParseError::NoError || !jsonDoc.isObject()) {
                diagnosisLabel->setText("Diagnosis: Invalid backend response");
                QMessageBox::warning(this, "Response Error", "Backend returned invalid JSON.");
                reply->deleteLater();
                return;
            }

            const QJsonObject obj = jsonDoc.object();
            if (obj.contains("error")) {
                diagnosisLabel->setText("Diagnosis: Backend error");
                QMessageBox::warning(this, "Prediction Error", obj.value("error").toString());
                reply->deleteLater();
                return;
            }

            const QString prediction = obj.value("prediction").toString("Unknown");
            const double confidence = obj.value("confidence").toDouble(0.0);
            const QString modelType = obj.value("model_type").toString("unknown");
            diagnosisLabel->setText(
                QString("Diagnosis: %1 | Confidence: %2% | Model: %3")
                    .arg(prediction)
                    .arg(QString::number(confidence * 100.0, 'f', 2))
                    .arg(modelType)
            );

            const QString camPath = obj.value("cam_image_path").toString();
            if (!camPath.isEmpty() && QFile::exists(camPath)) {
                camsLabel->setPixmap(QPixmap(camPath));
            } else {
                camsLabel->setText("CAMS Image\n(Not available)");
            }

            reply->deleteLater();
        });
    }
    
    void toggleTheme() {
        isDarkMode = !isDarkMode;
        settings->setValue("darkMode", isDarkMode);
        
        if (isDarkMode) {
            applyDarkMode();
        } else {
            applyLightMode();
        }
    }
    
    void applyLightMode() {
        themeBtn->setText("Dark Mode");
        
        QPalette lightPalette;
        lightPalette.setColor(QPalette::Window, QColor(255, 255, 255));
        lightPalette.setColor(QPalette::WindowText, QColor(0, 0, 0));
        lightPalette.setColor(QPalette::Base, QColor(245, 245, 245));
        lightPalette.setColor(QPalette::AlternateBase, QColor(235, 235, 235));
        lightPalette.setColor(QPalette::ToolTipBase, QColor(255, 255, 220));
        lightPalette.setColor(QPalette::ToolTipText, QColor(0, 0, 0));
        lightPalette.setColor(QPalette::Text, QColor(0, 0, 0));
        lightPalette.setColor(QPalette::Button, QColor(240, 240, 240));
        lightPalette.setColor(QPalette::ButtonText, QColor(0, 0, 0));
        lightPalette.setColor(QPalette::BrightText, QColor(255, 255, 255));
        lightPalette.setColor(QPalette::Link, QColor(0, 102, 204));
        lightPalette.setColor(QPalette::Highlight, QColor(76, 163, 224));
        lightPalette.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
        
        qApp->setPalette(lightPalette);
        
        QString lightStyle = R"(
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #e8e8e8;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d8d8d8;
            }
            QPushButton:pressed {
                background-color: #c8c8c8;
            }
            QLineEdit {
                background-color: #f5f5f5;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #000000;
            }
            QListWidget {
                background-color: #f5f5f5;
                color: #000000;
                border: 1px solid #cccccc;
            }
            QListWidget::item:selected {
                background-color: #4ca3e0;
                color: #ffffff;
            }
        )";
        qApp->setStyleSheet(lightStyle);
    }
    
    void applyDarkMode() {
        themeBtn->setText("Light Mode");
        
        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window, QColor(30, 30, 30));
        darkPalette.setColor(QPalette::WindowText, QColor(240, 240, 240));
        darkPalette.setColor(QPalette::Base, QColor(45, 45, 45));
        darkPalette.setColor(QPalette::AlternateBase, QColor(55, 55, 55));
        darkPalette.setColor(QPalette::ToolTipBase, QColor(30, 30, 30));
        darkPalette.setColor(QPalette::ToolTipText, QColor(240, 240, 240));
        darkPalette.setColor(QPalette::Text, QColor(240, 240, 240));
        darkPalette.setColor(QPalette::Button, QColor(50, 50, 50));
        darkPalette.setColor(QPalette::ButtonText, QColor(240, 240, 240));
        darkPalette.setColor(QPalette::BrightText, QColor(240, 240, 240));
        darkPalette.setColor(QPalette::Link, QColor(100, 180, 255));
        darkPalette.setColor(QPalette::Highlight, QColor(100, 150, 200));
        darkPalette.setColor(QPalette::HighlightedText, QColor(30, 30, 30));
        
        qApp->setPalette(darkPalette);
        
        QString darkStyle = R"(
            QWidget {
                background-color: #1e1e1e;
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #323232;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #424242;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #f0f0f0;
            }
            QListWidget {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #555555;
            }
            QListWidget::item:selected {
                background-color: #6496c8;
                color: #ffffff;
            }
        )";
        qApp->setStyleSheet(darkStyle);
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    AMD_GUI window;
    window.show();
    return app.exec();
}
