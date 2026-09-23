import QtQuick
import QtQuick.Controls

ComboBox {
    id: root
    font.pixelSize: 11
    implicitHeight: 32

    palette.window: "#151A22"
    palette.button: "#151A22"
    palette.buttonText: "#F0F4F8"
    palette.text: "#F0F4F8"
    palette.highlightedText: "#000000"
    palette.highlight: "#00E5FF"
    palette.base: "#151A22"
    palette.mid: "#2A2E35"

    background: Rectangle {
        implicitHeight: 32
        color: root.enabled ? "#151A22" : "#0B0E14"
        border.color: root.down || root.hovered ? "#00E5FF" : "#2A2E35"
        radius: 4
    }

    contentItem: Text {
        leftPadding: 10
        rightPadding: root.indicator.width + 12
        text: root.displayText
        font: root.font
        color: "#F0F4F8"
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    delegate: ItemDelegate {
        width: root.width
        highlighted: root.highlightedIndex === index

        background: Rectangle {
            color: highlighted ? "#00E5FF" : "#151A22"
        }

        contentItem: Text {
            text: modelData
            color: highlighted ? "#000000" : "#F0F4F8"
            font.pixelSize: 11
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
    }

    popup: Popup {
        y: root.height
        width: root.width
        implicitHeight: Math.min(420, contentItem.implicitHeight + 2)
        padding: 1

        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: root.popup.visible ? root.delegateModel : null
            currentIndex: root.highlightedIndex
            ScrollBar.vertical: ScrollBar {}
        }

        background: Rectangle {
            color: "#151A22"
            border.color: "#2A2E35"
            radius: 4
        }
    }
}
