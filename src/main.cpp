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

class AMD_GUI : public QWidget {
public:
    QLabel *fundusLabel;
    QLabel *camsLabel;
    QLabel *diagnosisLabel;
    QLineEdit *nameInput;
    QListWidget *historyList;
    QPushButton *themeBtn;
    bool isDarkMode;
    QSettings *settings;

    AMD_GUI() {
        setWindowTitle("AMD Detection System");
        resize(1000, 600);
        
        // Initialize settings for theme persistence
        settings = new QSettings("AMD_Detection", "AMD_GUI", this);
        isDarkMode = settings->value("darkMode", false).toBool();

        nameInput = new QLineEdit();
        nameInput->setPlaceholderText("Enter Patient Name");

        QPushButton *uploadBtn = new QPushButton("Upload Fundus Image");
        
        themeBtn = new QPushButton(isDarkMode ? "☀️ Light Mode" : "🌙 Dark Mode");
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
        connect(uploadBtn, &QPushButton::clicked, this, [=](){
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
            }
        });

        // History selection functionality
        connect(historyList, &QListWidget::itemClicked, this, [=](QListWidgetItem *item){
            fundusLabel->setPixmap(QPixmap(item->text()));
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
        themeBtn->setText("🌙 Dark Mode");
        
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
        themeBtn->setText("☀️ Light Mode");
        
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
